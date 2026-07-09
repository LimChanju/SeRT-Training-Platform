from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(V2_DIR)
PYTHON_PACKAGE_DIR = os.path.join(V2_DIR, ".python_packages")
ISAAC_TORCH_BUNDLE = os.environ.get(
    "ISAAC_TORCH_BUNDLE",
    os.path.expanduser("~/isaac-sim-4.5.0/exts/omni.isaac.ml_archive/pip_prebundle"),
)

for path in (PYTHON_PACKAGE_DIR, V2_DIR, PROJECT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BC policy from expert HDF5 trajectories.")
    parser.add_argument(
        "--data",
        default=os.path.join(V2_DIR, "trajectories", "expert_pick_place_v1.hdf5"),
        help="Input HDF5 trajectory file.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(V2_DIR, "policies", "bc_pick_place_v1.pt"),
        help="Output PyTorch checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--hidden-dims", default="256,256")
    parser.add_argument(
        "--target-dataset",
        default="expert_task_action",
        choices=("expert_task_action", "expert_target_action"),
        help="Dataset under each episode's actions group to imitate.",
    )
    parser.add_argument(
        "--install-missing-deps",
        action="store_true",
        help="Install h5py into v2/.python_packages/ if it is missing.",
    )
    return parser.parse_args()


def _ensure_training_deps(args: argparse.Namespace) -> None:
    try:
        import h5py  # noqa: F401
    except ModuleNotFoundError as exc:
        if not args.install_missing_deps:
            raise SystemExit(
                "Missing h5py. Rerun with --install-missing-deps, or install h5py "
                "into v2/.python_packages/."
            ) from exc
        import importlib
        import subprocess

        os.makedirs(PYTHON_PACKAGE_DIR, exist_ok=True)
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--target",
            PYTHON_PACKAGE_DIR,
            "--no-deps",
            "h5py>=3.8",
        ])
        importlib.invalidate_caches()
        import h5py  # noqa: F401

    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if os.path.isdir(ISAAC_TORCH_BUNDLE) and ISAAC_TORCH_BUNDLE not in sys.path:
            sys.path.insert(0, ISAAC_TORCH_BUNDLE)
        import torch  # noqa: F401

from rl import ACTION_DIM, ACTION_VERSION, OBSERVATION_VERSION  # noqa: E402
from rl.rewards import REWARD_VERSION  # noqa: E402


def _load_dataset(path: str, target_dataset: str) -> tuple[np.ndarray, np.ndarray, dict]:
    import h5py

    obs_chunks = []
    action_chunks = []
    metadata = {"episodes": 0, "transitions": 0}
    with h5py.File(path, "r") as h5:
        metadata.update({key: _attr_to_python(value) for key, value in h5.attrs.items()})
        episodes = h5["episodes"]
        for episode_name in sorted(episodes.keys()):
            group = episodes[episode_name]
            obs = np.asarray(group["obs_policy"], dtype=np.float32)
            action_path = f"actions/{target_dataset}"
            if action_path not in group:
                raise KeyError(f"{episode_name}: missing dataset '{action_path}'")
            actions = np.asarray(group[action_path], dtype=np.float32)
            if obs.shape[0] != actions.shape[0]:
                raise ValueError(
                    f"{episode_name}: obs/action length mismatch "
                    f"{obs.shape[0]} != {actions.shape[0]}"
                )
            obs_chunks.append(obs)
            action_chunks.append(actions)
        metadata["episodes"] = len(obs_chunks)
    if not obs_chunks:
        raise ValueError(f"No episodes found in {path}")
    obs_all = np.concatenate(obs_chunks, axis=0).astype(np.float32)
    action_all = np.concatenate(action_chunks, axis=0).astype(np.float32)
    metadata["transitions"] = int(obs_all.shape[0])
    metadata["target_dataset"] = target_dataset
    return obs_all, action_all, metadata


def _attr_to_python(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_hidden_dims(text: str) -> tuple[int, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        return (256, 256)
    return tuple(int(value) for value in values)


def _select_device(requested: str) -> torch.device:
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False")
    return torch.device(requested)


def _train(args: argparse.Namespace) -> None:
    _ensure_training_deps(args)
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    from rl.policies import MLPPolicy

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    obs_np, actions_np, data_meta = _load_dataset(args.data, args.target_dataset)
    obs_dim = int(obs_np.shape[1])
    if actions_np.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action dim {ACTION_DIM}, got {actions_np.shape[1]}")

    obs_mean = obs_np.mean(axis=0, keepdims=True).astype(np.float32)
    obs_std = obs_np.std(axis=0, keepdims=True).astype(np.float32)
    obs_std = np.maximum(obs_std, 1e-6)
    obs_norm = (obs_np - obs_mean) / obs_std

    obs_tensor = torch.from_numpy(obs_norm)
    action_tensor = torch.from_numpy(np.clip(actions_np, -1.0, 1.0))
    dataset = TensorDataset(obs_tensor, action_tensor)

    val_size = max(1, int(len(dataset) * float(args.val_ratio))) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )

    device = _select_device(args.device)
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    model = MLPPolicy(obs_dim, ACTION_DIM, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    history = []

    print(
        f"[TrainBC] data={args.data} transitions={len(dataset)} episodes={data_meta['episodes']} "
        f"target={args.target_dataset} obs_dim={obs_dim} device={device} torch={torch.__version__}"
    )
    if device.type == "cuda":
        print(f"[TrainBC] cuda={torch.cuda.get_device_name(0)}")

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for obs_batch, action_batch in train_loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            pred = model(obs_batch)
            loss = loss_fn(pred, action_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * obs_batch.shape[0]
            train_count += int(obs_batch.shape[0])
        train_loss = train_loss_sum / max(1, train_count)
        val_loss = _evaluate(model, val_loader, loss_fn, device) if val_loader is not None else train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[TrainBC] epoch={epoch:04d} train={train_loss:.6f} val={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "obs_mean": torch.from_numpy(obs_mean.squeeze(0)),
        "obs_std": torch.from_numpy(obs_std.squeeze(0)),
        "obs_dim": obs_dim,
        "action_dim": ACTION_DIM,
        "hidden_dims": hidden_dims,
        "observation_version": OBSERVATION_VERSION,
        "action_version": data_meta.get("action_version", ACTION_VERSION),
        "reward_version": REWARD_VERSION,
        "source_data": os.path.abspath(args.data),
        "data_metadata": data_meta,
        "train_args": vars(args),
        "history": history,
        "best_val_loss": best_val,
    }
    torch.save(checkpoint, args.output)
    history_path = os.path.splitext(args.output)[0] + "_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[TrainBC] saved checkpoint: {args.output}")
    print(f"[TrainBC] saved history: {history_path}")


def _evaluate(model, loader, loss_fn, device: torch.device) -> float:
    import torch

    if loader is None:
        return 0.0
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for obs_batch, action_batch in loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            loss = loss_fn(model(obs_batch), action_batch)
            loss_sum += float(loss.item()) * obs_batch.shape[0]
            count += int(obs_batch.shape[0])
    return loss_sum / max(1, count)


if __name__ == "__main__":
    _train(_parse_args())
