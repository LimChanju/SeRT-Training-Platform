from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(V2_DIR)
PYTHON_PACKAGE_DIR = os.path.join(V2_DIR, ".python_packages")
ISAAC_TORCH_BUNDLE = os.environ.get(
    "ISAAC_TORCH_BUNDLE",
    os.path.expanduser("~/isaac-sim-4.5.0/exts/omni.isaac.ml_archive/pip_prebundle"),
)

for path in (SCRIPT_DIR, PYTHON_PACKAGE_DIR, V2_DIR, PROJECT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


TARGET_VERSION = "expert_arm_joint_action_v0"
TARGET_DIM = 7


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BC policy from expert arm joint targets.")
    parser.add_argument(
        "--data",
        default=os.path.join("v2", "trajectories", "expert_pick_place_v0_100eps.hdf5"),
        help="Input HDF5 trajectory file.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("v2", "policies", "bc_arm_joint_v0.pt"),
        help="Output PyTorch checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--hidden-dims", default="256,256")
    parser.add_argument(
        "--install-missing-deps",
        action="store_true",
        help="Install h5py into v2/.python_packages/ if it is missing.",
    )
    return parser.parse_args()


def _ensure_training_deps(args: argparse.Namespace) -> None:
    try:
        import h5py  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
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


from observations import OBSERVATION_DIM, OBSERVATION_VERSION  # noqa: E402


def _load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    import h5py

    obs_chunks = []
    target_chunks = []
    metadata = {"episodes": 0, "transitions": 0, "kept_transitions": 0, "dropped_nonfinite": 0}
    with h5py.File(path, "r") as h5:
        metadata.update({key: _attr_to_python(value) for key, value in h5.attrs.items()})
        episodes = h5["episodes"]
        for episode_name in sorted(episodes.keys()):
            group = episodes[episode_name]
            obs = np.asarray(group["obs_policy"], dtype=np.float32)
            joint = np.asarray(group["actions/expert_joint_action"], dtype=np.float32)[:, :TARGET_DIM]
            if obs.shape[0] != joint.shape[0]:
                raise ValueError(
                    f"{episode_name}: obs/joint length mismatch {obs.shape[0]} != {joint.shape[0]}"
                )
            finite = np.isfinite(obs).all(axis=1) & np.isfinite(joint).all(axis=1)
            metadata["dropped_nonfinite"] += int((~finite).sum())
            obs_chunks.append(obs[finite])
            target_chunks.append(joint[finite])
        metadata["episodes"] = len(obs_chunks)
    if not obs_chunks:
        raise ValueError(f"No episodes found in {path}")
    obs_all = np.concatenate(obs_chunks, axis=0).astype(np.float32)
    target_all = np.concatenate(target_chunks, axis=0).astype(np.float32)
    if obs_all.shape[0] == 0:
        raise ValueError(f"No finite joint-action transitions found in {path}")
    metadata["kept_transitions"] = int(obs_all.shape[0])
    metadata["transitions"] = int(obs_all.shape[0] + metadata["dropped_nonfinite"])
    return obs_all, target_all, metadata


def _attr_to_python(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _parse_hidden_dims(text: str) -> tuple[int, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        return (256, 256)
    return tuple(int(value) for value in values)


def _select_device(requested: str):
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

    from policies import MLPRegressor

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    obs_np, target_np, data_meta = _load_dataset(args.data)
    if obs_np.shape[1] != OBSERVATION_DIM:
        raise ValueError(f"Expected obs dim {OBSERVATION_DIM}, got {obs_np.shape[1]}")
    if target_np.shape[1] != TARGET_DIM:
        raise ValueError(f"Expected target dim {TARGET_DIM}, got {target_np.shape[1]}")

    obs_mean = obs_np.mean(axis=0, keepdims=True).astype(np.float32)
    obs_std = np.maximum(obs_np.std(axis=0, keepdims=True).astype(np.float32), 1e-6)
    target_mean = target_np.mean(axis=0, keepdims=True).astype(np.float32)
    target_std = np.maximum(target_np.std(axis=0, keepdims=True).astype(np.float32), 1e-6)
    target_min = target_np.min(axis=0, keepdims=True).astype(np.float32)
    target_max = target_np.max(axis=0, keepdims=True).astype(np.float32)
    obs_norm = (obs_np - obs_mean) / obs_std
    target_norm = (target_np - target_mean) / target_std

    dataset = TensorDataset(torch.from_numpy(obs_norm), torch.from_numpy(target_norm))
    val_size = max(1, int(len(dataset) * float(args.val_ratio))) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )

    device = _select_device(args.device)
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    model = MLPRegressor(OBSERVATION_DIM, TARGET_DIM, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    history = []

    print(
        f"[TrainJointBC] data={args.data} kept={data_meta['kept_transitions']} "
        f"dropped_nonfinite={data_meta['dropped_nonfinite']} episodes={data_meta['episodes']} "
        f"device={device} torch={torch.__version__}"
    )
    if device.type == "cuda":
        print(f"[TrainJointBC] cuda={torch.cuda.get_device_name(0)}")

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for obs_batch, target_batch in train_loader:
            obs_batch = obs_batch.to(device)
            target_batch = target_batch.to(device)
            pred = model(obs_batch)
            loss = loss_fn(pred, target_batch)
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
            print(f"[TrainJointBC] epoch={epoch:04d} train={train_loss:.6f} val={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "model_class": "MLPRegressor",
        "target_version": TARGET_VERSION,
        "obs_mean": torch.from_numpy(obs_mean.squeeze(0)),
        "obs_std": torch.from_numpy(obs_std.squeeze(0)),
        "target_mean": torch.from_numpy(target_mean.squeeze(0)),
        "target_std": torch.from_numpy(target_std.squeeze(0)),
        "target_min": torch.from_numpy(target_min.squeeze(0)),
        "target_max": torch.from_numpy(target_max.squeeze(0)),
        "obs_dim": OBSERVATION_DIM,
        "action_dim": TARGET_DIM,
        "hidden_dims": hidden_dims,
        "observation_version": OBSERVATION_VERSION,
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
    print(f"[TrainJointBC] saved checkpoint: {args.output}")
    print(f"[TrainJointBC] saved history: {history_path}")


def _evaluate(model, loader, loss_fn, device) -> float:
    import torch

    if loader is None:
        return 0.0
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for obs_batch, target_batch in loader:
            obs_batch = obs_batch.to(device)
            target_batch = target_batch.to(device)
            loss = loss_fn(model(obs_batch), target_batch)
            loss_sum += float(loss.item()) * obs_batch.shape[0]
            count += int(obs_batch.shape[0])
    return loss_sum / max(1, count)


if __name__ == "__main__":
    _train(_parse_args())
