from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RL_DIR = os.path.join(SCRIPT_DIR, "rl")
PYTHON_PACKAGE_DIR = os.path.join(SCRIPT_DIR, ".python_packages")
ISAAC_TORCH_BUNDLE = os.environ.get(
    "ISAAC_TORCH_BUNDLE",
    os.path.expanduser("~/isaac-sim-4.5.0/exts/omni.isaac.ml_archive/pip_prebundle"),
)

for path in (RL_DIR, PYTHON_PACKAGE_DIR, SCRIPT_DIR, PROJECT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small HRI risk / pseudo-ErrP reward model from trajectory HDF5."
    )
    parser.add_argument(
        "--data",
        default=os.path.join(SCRIPT_DIR, "trajectories", "hri_vr_expert_v0.hdf5"),
        help="HDF5 trajectory containing obs/human fields.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(SCRIPT_DIR, "policies", "risk_model_hri_v0.pt"),
        help="Path to save the PyTorch risk model checkpoint.",
    )
    parser.add_argument(
        "--history",
        default="",
        help="Optional JSON history path. Empty uses '<output stem>_history.json'.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dims", default="128,128")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--collision-dist",
        type=float,
        default=0.03,
        help="Distance that maps to risk 1.0.",
    )
    parser.add_argument(
        "--safe-dist",
        type=float,
        default=0.12,
        help="Distance that maps to risk 0.0.",
    )
    parser.add_argument(
        "--augment-near-hand",
        type=int,
        default=4,
        help=(
            "Synthetic near-hand samples per real sample. Use 0 for pure logged data. "
            "Needed when the HRI file has no risky frames."
        ),
    )
    parser.add_argument(
        "--max-real-samples",
        type=int,
        default=0,
        help="Optional cap for real samples before augmentation. 0 uses all.",
    )
    return parser.parse_args()


def _ensure_torch() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if os.path.isdir(ISAAC_TORCH_BUNDLE) and ISAAC_TORCH_BUNDLE not in sys.path:
            sys.path.insert(0, ISAAC_TORCH_BUNDLE)
        import torch  # noqa: F401


_ensure_torch()

import h5py  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from rl.risk_model import RISK_FEATURE_NAMES, RISK_MODEL_VERSION, RiskMLP  # noqa: E402


def _train(args: argparse.Namespace) -> None:
    started_at = time.time()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _select_device(args.device)
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    data_path = _resolve_project_path(args.data)
    output_path = _resolve_output_path(args.output)
    history_path = _resolve_history_path(args.history, output_path)

    features, labels, meta = _load_dataset(
        data_path,
        collision_dist=float(args.collision_dist),
        safe_dist=float(args.safe_dist),
        rng=rng,
        augment_near_hand=max(0, int(args.augment_near_hand)),
        max_real_samples=max(0, int(args.max_real_samples)),
    )
    if features.shape[0] < 2:
        raise RuntimeError("Risk model dataset is too small.")

    order = rng.permutation(features.shape[0])
    features = features[order]
    labels = labels[order]
    val_count = int(round(features.shape[0] * float(np.clip(args.val_ratio, 0.05, 0.5))))
    val_count = min(max(1, val_count), features.shape[0] - 1)
    x_val = features[:val_count]
    y_val = labels[:val_count]
    x_train = features[val_count:]
    y_train = labels[val_count:]

    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std

    model = RiskMLP(features.shape[1], hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    criterion = nn.BCEWithLogitsLoss()

    x_train_t = torch.from_numpy(x_train_n).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    x_val_t = torch.from_numpy(x_val_n).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    print(
        f"[TrainRisk] data={data_path} samples={features.shape[0]} "
        f"real={meta['real_samples']} synthetic={meta['synthetic_samples']} "
        f"feature_dim={features.shape[1]} device={device} torch={torch.__version__}",
        flush=True,
    )
    print(
        f"[TrainRisk] label mean={float(labels.mean()):.4f} "
        f"positive>0.5={int(np.sum(labels > 0.5))}/{labels.shape[0]} "
        f"safe_dist={args.safe_dist} collision_dist={args.collision_dist}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"[TrainRisk] cuda={torch.cuda.get_device_name(0)}", flush=True)

    history: list[dict[str, float]] = []
    batch_size = max(1, int(args.batch_size))
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_order = torch.randperm(x_train_t.shape[0], device=device)
        train_losses = []
        for start in range(0, x_train_t.shape[0], batch_size):
            idx = train_order[start : start + batch_size]
            logits = model(x_train_t[idx])
            loss = criterion(logits, y_train_t[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        metrics = _evaluate(model, criterion, x_val_t, y_val_t)
        row = {
            "epoch": float(epoch),
            "train_bce": float(np.mean(train_losses)),
            **metrics,
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(args.epochs):
            print(
                f"[TrainRisk] epoch={epoch:04d} train_bce={row['train_bce']:.6f} "
                f"val_bce={row['val_bce']:.6f} val_mae={row['val_mae']:.6f} "
                f"val_mse={row['val_mse']:.6f}",
                flush=True,
            )

    checkpoint = {
        "model_version": RISK_MODEL_VERSION,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "feature_names": RISK_FEATURE_NAMES,
        "feature_dim": int(features.shape[1]),
        "hidden_dims": hidden_dims,
        "feature_mean": torch.from_numpy(mean.squeeze(0)),
        "feature_std": torch.from_numpy(std.squeeze(0)),
        "collision_dist_m": float(args.collision_dist),
        "safe_dist_m": float(args.safe_dist),
        "source_data": data_path,
        "dataset_meta": meta,
        "train_args": vars(args),
        "history": history,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)
    _write_json(
        history_path,
        {
            "created_unix": time.time(),
            "duration_sec": time.time() - started_at,
            "checkpoint": output_path,
            "dataset_meta": meta,
            "history": history,
        },
    )
    print(f"[TrainRisk] saved checkpoint: {output_path}", flush=True)
    print(f"[TrainRisk] saved history: {history_path}", flush=True)


def _load_dataset(
    path: str,
    *,
    collision_dist: float,
    safe_dist: float,
    rng: np.random.Generator,
    augment_near_hand: int,
    max_real_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    real_features = []
    real_labels = []
    real_min_dists = []
    with h5py.File(path, "r") as h5:
        if "episodes" not in h5:
            raise KeyError(f"No episodes group in {path}")
        episode_names = sorted(h5["episodes"].keys())
        for episode_name in episode_names:
            episode = h5["episodes"][episode_name]
            obs = episode["obs"]
            features = _features_from_obs_group(obs)
            labels = _labels_from_episode(
                episode,
                collision_dist=collision_dist,
                safe_dist=safe_dist,
            )
            real_features.append(features)
            real_labels.append(labels)
            real_min_dists.append(np.asarray(obs["min_hand_gripper_dist"], dtype=np.float32).reshape(-1))

    if not real_features:
        raise ValueError(f"No episodes found in {path}")
    x_real = np.concatenate(real_features, axis=0).astype(np.float32)
    y_real = np.concatenate(real_labels, axis=0).astype(np.float32)
    min_dists = np.concatenate(real_min_dists, axis=0).astype(np.float32)
    if max_real_samples > 0 and x_real.shape[0] > max_real_samples:
        idx = rng.choice(x_real.shape[0], size=max_real_samples, replace=False)
        x_real = x_real[idx]
        y_real = y_real[idx]
        min_dists = min_dists[idx]

    synthetic = []
    synthetic_labels = []
    if augment_near_hand > 0:
        synthetic, synthetic_labels = _augment_near_hand_samples(
            x_real,
            rng=rng,
            per_sample=augment_near_hand,
            collision_dist=collision_dist,
            safe_dist=safe_dist,
        )

    if len(synthetic):
        x = np.concatenate([x_real, synthetic], axis=0).astype(np.float32)
        y = np.concatenate([y_real, synthetic_labels], axis=0).astype(np.float32)
    else:
        x = x_real.astype(np.float32)
        y = y_real.astype(np.float32)

    meta = {
        "path": path,
        "real_samples": int(x_real.shape[0]),
        "synthetic_samples": int(0 if not len(synthetic) else synthetic.shape[0]),
        "real_label_mean": float(y_real.mean()),
        "real_positive_gt_0_5": int(np.sum(y_real > 0.5)),
        "real_min_hand_dist_min": float(np.min(min_dists)),
        "real_min_hand_dist_mean": float(np.mean(min_dists)),
        "total_label_mean": float(y.mean()),
        "total_positive_gt_0_5": int(np.sum(y > 0.5)),
    }
    return x, y, meta


def _features_from_obs_group(obs_group) -> np.ndarray:
    parts = []
    for name in RISK_FEATURE_NAMES:
        if name not in obs_group:
            raise KeyError(f"Observation group is missing '{name}'")
        arr = np.asarray(obs_group[name], dtype=np.float32)
        parts.append(arr.reshape(arr.shape[0], -1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def _labels_from_episode(
    episode,
    *,
    collision_dist: float,
    safe_dist: float,
) -> np.ndarray:
    obs = episode["obs"]
    min_dist = np.asarray(obs["min_hand_gripper_dist"], dtype=np.float32).reshape(-1)
    distance_risk = _distance_to_risk(min_dist, collision_dist=collision_dist, safe_dist=safe_dist)
    collision = np.asarray(obs["human_robot_collision"], dtype=np.float32).reshape(-1)
    near = np.asarray(obs["near_human"], dtype=np.float32).reshape(-1) * 0.35
    label = np.maximum(distance_risk, np.maximum(collision, near))
    if "errp" in episode and "feedback" in episode["errp"]:
        feedback = np.asarray(episode["errp"]["feedback"], dtype=np.float32).reshape(-1)
        if feedback.shape[0] == label.shape[0]:
            label = np.maximum(label, np.clip(feedback, 0.0, 1.0))
    return np.clip(label, 0.0, 1.0).astype(np.float32)


def _augment_near_hand_samples(
    x_real: np.ndarray,
    *,
    rng: np.random.Generator,
    per_sample: int,
    collision_dist: float,
    safe_dist: float,
) -> tuple[np.ndarray, np.ndarray]:
    slices = _feature_slices()
    samples = []
    labels = []
    for row in x_real:
        ee_pos = row[slices["ee_pos"]]
        base = np.array(row, copy=True)
        for _ in range(per_sample):
            augmented = np.array(base, copy=True)
            dist = float(rng.uniform(collision_dist * 0.5, safe_dist * 1.15))
            direction = rng.normal(size=3)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-8:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                direction = direction / norm
            hand_pos = ee_pos + direction * dist
            use_left = bool(rng.random() < 0.5)
            if use_left:
                augmented[slices["human_left_hand_pos"]] = hand_pos
                augmented[slices["ee_to_left_hand"]] = hand_pos - ee_pos
            else:
                augmented[slices["human_right_hand_pos"]] = hand_pos
                augmented[slices["ee_to_right_hand"]] = hand_pos - ee_pos
            augmented[slices["min_hand_gripper_dist"]] = dist
            samples.append(augmented)
            labels.append(_distance_to_risk(np.array([dist]), collision_dist=collision_dist, safe_dist=safe_dist)[0])
    if not samples:
        return np.zeros((0, x_real.shape[1]), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.stack(samples).astype(np.float32), np.asarray(labels, dtype=np.float32)


def _distance_to_risk(
    distances: np.ndarray,
    *,
    collision_dist: float,
    safe_dist: float,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float32)
    denom = max(float(safe_dist) - float(collision_dist), 1e-6)
    risk = (float(safe_dist) - distances) / denom
    return np.clip(risk, 0.0, 1.0).astype(np.float32)


def _feature_slices() -> dict[str, slice]:
    sizes = {
        "robot_joint_pos": 7,
        "gripper_width": 1,
        "ee_pos": 3,
        "cube_pos": 3,
        "place_target_pos": 3,
        "ee_to_cube": 3,
        "cube_to_place_target": 3,
        "task_phase": 4,
        "controller_event": 10,
        "controller_t": 1,
        "human_head_pos": 3,
        "human_left_hand_pos": 3,
        "human_right_hand_pos": 3,
        "ee_to_left_hand": 3,
        "ee_to_right_hand": 3,
        "min_hand_gripper_dist": 1,
    }
    slices = {}
    cursor = 0
    for name in RISK_FEATURE_NAMES:
        dim = sizes[name]
        slices[name] = slice(cursor, cursor + dim)
        cursor += dim
    return slices


@torch.no_grad()
def _evaluate(
    model: RiskMLP,
    criterion: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    logits = model(features)
    loss = criterion(logits, labels)
    probs = torch.sigmoid(logits)
    mae = torch.mean(torch.abs(probs - labels))
    mse = torch.mean((probs - labels) ** 2)
    return {
        "val_bce": float(loss.detach().cpu()),
        "val_mae": float(mae.detach().cpu()),
        "val_mse": float(mse.detach().cpu()),
    }


def _select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("[TrainRisk] CUDA requested but unavailable; falling back to CPU.", flush=True)
        name = "cpu"
    return torch.device(name)


def _parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    return dims or (128, 128)


def _resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_DIR, path))


def _resolve_output_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_DIR, path))


def _resolve_history_path(path: str, output_path: str) -> str:
    if path:
        return _resolve_output_path(path)
    root, _ = os.path.splitext(output_path)
    return root + "_history.json"


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    _train(_parse_args())
