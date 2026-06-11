from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a BC policy on expert HDF5 trajectories.")
    parser.add_argument(
        "--data",
        default=os.path.join("v2", "trajectories", "expert_pick_place_v0.hdf5"),
        help="Input HDF5 trajectory file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("v2", "policies", "bc_pick_place_v0.pt"),
        help="PyTorch checkpoint produced by train_bc.py.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=8, help="Number of prediction examples to print.")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save evaluation metrics as JSON.",
    )
    parser.add_argument(
        "--install-missing-deps",
        action="store_true",
        help="Install h5py into v2/.python_packages/ if it is missing.",
    )
    return parser.parse_args()


def _ensure_eval_deps(args: argparse.Namespace) -> None:
    try:
        import h5py  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        if not args.install_missing_deps:
            raise SystemExit(
                "Missing h5py. Rerun with --install-missing-deps, or first run "
                "collect_expert_trajectories.py/train_bc.py with --install-missing-deps."
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


from actions import ACTION_DIM, ACTION_NAMES  # noqa: E402
from observations import OBSERVATION_DIM  # noqa: E402


def _load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import h5py

    obs_chunks = []
    action_chunks = []
    episode_lengths = {}
    metadata: dict[str, Any] = {"episodes": 0, "transitions": 0}
    with h5py.File(path, "r") as h5:
        metadata.update({key: _attr_to_python(value) for key, value in h5.attrs.items()})
        episodes = h5["episodes"]
        for episode_name in sorted(episodes.keys()):
            group = episodes[episode_name]
            obs = np.asarray(group["obs_policy"], dtype=np.float32)
            actions = np.asarray(group["actions/expert_task_action"], dtype=np.float32)
            if obs.shape[0] != actions.shape[0]:
                raise ValueError(
                    f"{episode_name}: obs/action length mismatch "
                    f"{obs.shape[0]} != {actions.shape[0]}"
                )
            obs_chunks.append(obs)
            action_chunks.append(actions)
            episode_lengths[episode_name] = int(obs.shape[0])
        metadata["episodes"] = len(obs_chunks)
    if not obs_chunks:
        raise ValueError(f"No episodes found in {path}")
    obs_all = np.concatenate(obs_chunks, axis=0).astype(np.float32)
    action_all = np.concatenate(action_chunks, axis=0).astype(np.float32)
    metadata["transitions"] = int(obs_all.shape[0])
    metadata["episode_lengths"] = episode_lengths
    return obs_all, action_all, metadata


def _attr_to_python(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _select_device(requested: str):
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False")
    return torch.device(requested)


def _torch_load(path: str, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _predict(model, obs_norm: np.ndarray, device, batch_size: int) -> np.ndarray:
    import torch

    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, obs_norm.shape[0], batch_size):
            end = min(start + batch_size, obs_norm.shape[0])
            obs_batch = torch.from_numpy(obs_norm[start:end]).to(device)
            preds.append(model(obs_batch).detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def _evaluate(args: argparse.Namespace) -> None:
    _ensure_eval_deps(args)
    import torch

    from policies import MLPPolicy

    obs_np, action_np, data_meta = _load_dataset(args.data)
    if obs_np.shape[1] != OBSERVATION_DIM:
        raise ValueError(f"Expected obs dim {OBSERVATION_DIM}, got {obs_np.shape[1]}")
    if action_np.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action dim {ACTION_DIM}, got {action_np.shape[1]}")

    device = _select_device(args.device)
    checkpoint = _torch_load(args.checkpoint, device)
    obs_mean = _tensor_to_numpy(checkpoint["obs_mean"]).reshape(1, -1)
    obs_std = _tensor_to_numpy(checkpoint["obs_std"]).reshape(1, -1)
    hidden_dims = tuple(int(value) for value in checkpoint.get("hidden_dims", (256, 256)))
    model = MLPPolicy(
        int(checkpoint.get("obs_dim", OBSERVATION_DIM)),
        int(checkpoint.get("action_dim", ACTION_DIM)),
        hidden_dims=hidden_dims,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    obs_norm = (obs_np - obs_mean) / np.maximum(obs_std, 1e-6)
    pred_np = _predict(model, obs_norm.astype(np.float32), device, max(1, args.batch_size))
    target_np = np.clip(action_np, -1.0, 1.0)
    error = pred_np - target_np
    abs_error = np.abs(error)
    sq_error = error * error

    metrics = {
        "data": os.path.abspath(args.data),
        "checkpoint": os.path.abspath(args.checkpoint),
        "device": str(device),
        "torch_version": torch.__version__,
        "episodes": int(data_meta["episodes"]),
        "transitions": int(data_meta["transitions"]),
        "mse": float(np.mean(sq_error)),
        "mae": float(np.mean(abs_error)),
        "max_abs_error": float(np.max(abs_error)),
        "per_action": {
            name: {
                "mse": float(np.mean(sq_error[:, idx])),
                "mae": float(np.mean(abs_error[:, idx])),
                "max_abs_error": float(np.max(abs_error[:, idx])),
            }
            for idx, name in enumerate(ACTION_NAMES)
        },
        "dataset_metadata": data_meta,
        "checkpoint_metadata": {
            "observation_version": checkpoint.get("observation_version", ""),
            "action_version": checkpoint.get("action_version", ""),
            "reward_version": checkpoint.get("reward_version", ""),
            "best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
            "hidden_dims": list(hidden_dims),
        },
    }

    print(
        f"[EvalBC] data={args.data} checkpoint={args.checkpoint} "
        f"transitions={metrics['transitions']} episodes={metrics['episodes']} "
        f"device={device} torch={torch.__version__}"
    )
    if device.type == "cuda":
        print(f"[EvalBC] cuda={torch.cuda.get_device_name(0)}")
    print(
        f"[EvalBC] overall mse={metrics['mse']:.6f} "
        f"mae={metrics['mae']:.6f} max_abs={metrics['max_abs_error']:.6f}"
    )
    for name in ACTION_NAMES:
        dim_metrics = metrics["per_action"][name]
        print(
            f"[EvalBC] {name:>11s} mse={dim_metrics['mse']:.6f} "
            f"mae={dim_metrics['mae']:.6f} max_abs={dim_metrics['max_abs_error']:.6f}"
        )

    _print_samples(pred_np, target_np, abs_error, max(0, args.samples))

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[EvalBC] saved metrics: {args.output_json}")


def _print_samples(pred: np.ndarray, target: np.ndarray, abs_error: np.ndarray, sample_count: int) -> None:
    if sample_count <= 0:
        return
    count = min(sample_count, pred.shape[0])
    indices = np.linspace(0, pred.shape[0] - 1, num=count, dtype=int)
    print("[EvalBC] sample predictions:")
    for idx in indices:
        pred_text = np.array2string(pred[idx], precision=3, suppress_small=True)
        target_text = np.array2string(target[idx], precision=3, suppress_small=True)
        err_text = np.array2string(abs_error[idx], precision=3, suppress_small=True)
        print(f"  idx={idx:05d} pred={pred_text} expert={target_text} abs_err={err_text}")


if __name__ == "__main__":
    _evaluate(_parse_args())
