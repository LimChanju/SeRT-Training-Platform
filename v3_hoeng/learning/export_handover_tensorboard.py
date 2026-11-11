from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET_DIR = Path("data/trajectories/handover_dataset_v0")
DEFAULT_LOG_DIR = Path("runs/handover_dataset_v0")
METRIC_KEYS = (
    "grasp_metrics_center_dist",
    "grasp_metrics_min_dist",
    "grasp_metrics_width",
    "grasp_metrics_jaw_offset",
    "grasp_metrics_vertical_offset",
    "grasp_metrics_depth_offset",
    "grasp_metrics_head_dist",
)
ACTION_NAMES = ("dx", "dy", "dz", "qw", "qx", "qy", "qz", "gripper")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export existing handover NPZ episodes to TensorBoard event logs."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_DATASET_DIR),
        help="Dataset directory or one episode_*.npz file.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help="TensorBoard log directory to write.",
    )
    parser.add_argument(
        "--success-dist",
        type=float,
        default=0.08,
        help="Maximum final cube-place distance considered quality-ok.",
    )
    parser.add_argument(
        "--step-stride",
        type=int,
        default=1,
        help="Write every Nth timestep for per-episode scalar views.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Write only aggregate per-episode scalars, not per-step episode runs.",
    )
    parser.add_argument(
        "--include-joints",
        action="store_true",
        help="Also export joint_pos and joint_action per-step scalars.",
    )
    parser.add_argument(
        "--include-observations",
        action="store_true",
        help="Also export each obs_state dimension per-step. This can create many tags.",
    )
    args = parser.parse_args()

    input_path = Path(args.path).expanduser()
    episodes = _episode_paths(input_path)
    if not episodes:
        raise SystemExit(f"No episode_*.npz files found at {input_path}")

    writer_cls = _load_summary_writer()
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    stage_maps = _collect_stage_maps(episodes)
    summaries = []

    dataset_writer = writer_cls(str(log_dir / "dataset"))
    try:
        for episode_index, episode_path in enumerate(episodes):
            summary = _episode_summary(episode_path, success_dist_m=float(args.success_dist))
            summaries.append(summary)
            _write_dataset_scalars(dataset_writer, summary, episode_index)
            if not args.summary_only:
                episode_writer = writer_cls(str(log_dir / "episodes" / episode_path.stem))
                try:
                    _write_episode_run(
                        episode_writer,
                        episode_path,
                        summary=summary,
                        stage_maps=stage_maps,
                        step_stride=max(1, int(args.step_stride)),
                        include_joints=bool(args.include_joints),
                        include_observations=bool(args.include_observations),
                    )
                finally:
                    episode_writer.close()

        dataset_summary = _dataset_summary(summaries)
        _write_dataset_summary(dataset_writer, dataset_summary, summaries, stage_maps)
    finally:
        dataset_writer.close()

    summary_path = log_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[TensorBoard] exported {len(episodes)} episode(s) to: {log_dir}")
    print(f"[TensorBoard] wrote summary JSON: {summary_path}")
    print(f"[TensorBoard] view with: tensorboard --logdir {log_dir}")


def _load_summary_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(
            "TensorBoard export requires PyTorch's SummaryWriter and the tensorboard package.\n"
            "Install project dependencies in the Isaac Python environment, for example:\n"
            "  /home/railab/isaac-sim-4.5.0/python.sh -m pip install -r requirements.txt"
        ) from exc
    return SummaryWriter


def _episode_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("episode_*.npz"))


def _episode_summary(episode_path: Path, *, success_dist_m: float) -> dict[str, Any]:
    data = np.load(episode_path, allow_pickle=False)
    length = _length(data)
    cube_pos = _array(data, "cube_pos", (length, 3))
    place_pos = _array(data, "place_pos", (length, 3))
    action = _array(data, "action", (length, 8))
    sim_time = _series(data, "sim_time", length)
    robot_has_taken_over = _bool_series(data, "robot_has_taken_over", length)
    final_dist = _final_dist(cube_pos, place_pos)
    success = _scalar_bool(data, "success")
    quality_warnings = []
    if success and np.isfinite(final_dist) and final_dist > success_dist_m:
        quality_warnings.append(
            f"success=True but final cube-place dist {final_dist:.4f} m > {success_dist_m:.4f} m"
        )
    if length and not np.any(action[:, 7] < -0.5):
        quality_warnings.append("no close gripper command found")
    if length and not np.any(action[:, 7] > 0.5):
        quality_warnings.append("no open gripper command found")

    return {
        "episode": episode_path.stem,
        "path": str(episode_path),
        "length": int(length),
        "duration_s": float(sim_time[-1] - sim_time[0]) if length else 0.0,
        "success": success,
        "quality_ok": len(quality_warnings) == 0,
        "quality_warnings": quality_warnings,
        "failure_reason": _scalar_str(data, "failure_reason"),
        "instruction": _scalar_str(data, "instruction"),
        "active_hand": _first_nonempty(data, "active_hand"),
        "active_cube": _first_nonempty(data, "active_cube"),
        "final_cube_place_dist_m": final_dist,
        "takeover_step_index": _first_true_index(robot_has_taken_over),
        "phase_counts": _value_counts(data, "phase"),
        "oriented_stage_counts": _value_counts(data, "oriented_stage"),
        "physical_carry_stage_counts": _value_counts(data, "physical_carry_stage"),
        "gripper_open_steps": int(np.sum(action[:, 7] > 0.5)) if length else 0,
        "gripper_close_steps": int(np.sum(action[:, 7] < -0.5)) if length else 0,
        "gripper_neutral_steps": int(np.sum(np.abs(action[:, 7]) <= 0.5)) if length else 0,
        "action_delta_stats": {
            name: _finite_stats(action[:, idx]) if length else _nan_stats()
            for idx, name in enumerate(("dx", "dy", "dz"))
        },
    }


def _write_dataset_scalars(writer, summary: dict[str, Any], episode_index: int) -> None:
    writer.add_scalar("episode/length_steps", summary["length"], episode_index)
    writer.add_scalar("episode/duration_s", summary["duration_s"], episode_index)
    writer.add_scalar("episode/success", 1.0 if summary["success"] else 0.0, episode_index)
    writer.add_scalar("episode/quality_ok", 1.0 if summary["quality_ok"] else 0.0, episode_index)
    writer.add_scalar(
        "episode/final_cube_place_dist_m",
        _finite_or_zero(summary["final_cube_place_dist_m"]),
        episode_index,
    )
    writer.add_scalar(
        "episode/takeover_step_index",
        -1 if summary["takeover_step_index"] is None else summary["takeover_step_index"],
        episode_index,
    )
    writer.add_scalar("episode/gripper_open_steps", summary["gripper_open_steps"], episode_index)
    writer.add_scalar("episode/gripper_close_steps", summary["gripper_close_steps"], episode_index)
    writer.add_scalar("episode/gripper_neutral_steps", summary["gripper_neutral_steps"], episode_index)
    for axis, stats in summary["action_delta_stats"].items():
        for name, value in stats.items():
            writer.add_scalar(f"episode_action/{axis}_{name}", _finite_or_zero(value), episode_index)
    for key, counts in (
        ("phase_count", summary["phase_counts"]),
        ("oriented_stage_count", summary["oriented_stage_counts"]),
        ("physical_carry_stage_count", summary["physical_carry_stage_counts"]),
    ):
        for label, count in counts.items():
            writer.add_scalar(f"{key}/{_safe_tag(label)}", count, episode_index)


def _write_episode_run(
    writer,
    episode_path: Path,
    *,
    summary: dict[str, Any],
    stage_maps: dict[str, dict[str, int]],
    step_stride: int,
    include_joints: bool,
    include_observations: bool,
) -> None:
    data = np.load(episode_path, allow_pickle=False)
    length = _length(data)
    cube_pos = _array(data, "cube_pos", (length, 3))
    place_pos = _array(data, "place_pos", (length, 3))
    target_ee_pos = _array(data, "target_ee_pos", (length, 3))
    action = _array(data, "action", (length, 8))
    sim_time = _series(data, "sim_time", length)
    robot_has_taken_over = _bool_series(data, "robot_has_taken_over", length)
    phases = _string_series(data, "phase", length)
    oriented_stages = _string_series(data, "oriented_stage", length)
    carry_stages = _string_series(data, "physical_carry_stage", length)
    joint_pos = _array(data, "joint_pos", (length, 9)) if include_joints else None
    joint_action = _array(data, "joint_action", (length, 9)) if include_joints else None
    obs_state = _array(data, "obs_state", (length, 0)) if include_observations else None

    writer.add_text("episode/metadata", _metadata_markdown(summary, stage_maps), 0)
    for i in range(0, length, step_stride):
        step = i
        elapsed = float(sim_time[i] - sim_time[0]) if length else 0.0
        writer.add_scalar("time/elapsed_s", elapsed, step)
        writer.add_scalar("state/robot_has_taken_over", int(robot_has_taken_over[i]), step)
        writer.add_scalar(
            "stage/phase_id",
            _label_id(stage_maps["phase"], phases[i]),
            step,
        )
        writer.add_scalar(
            "stage/oriented_stage_id",
            _label_id(stage_maps["oriented_stage"], oriented_stages[i]),
            step,
        )
        writer.add_scalar(
            "stage/physical_carry_stage_id",
            _label_id(stage_maps["physical_carry_stage"], carry_stages[i]),
            step,
        )
        _write_vec3(writer, "position/cube", cube_pos[i], step)
        _write_vec3(writer, "position/place", place_pos[i], step)
        _write_vec3(writer, "position/target_ee", target_ee_pos[i], step)
        writer.add_scalar(
            "distance/cube_to_place_m",
            _norm_or_zero(cube_pos[i] - place_pos[i]),
            step,
        )
        writer.add_scalar(
            "distance/target_ee_to_cube_m",
            _norm_or_zero(target_ee_pos[i] - cube_pos[i]),
            step,
        )
        for idx, name in enumerate(ACTION_NAMES):
            writer.add_scalar(f"action/{name}", _finite_or_zero(action[i, idx]), step)
        for key in METRIC_KEYS:
            values = _series(data, key, length)
            if i < values.size:
                writer.add_scalar(f"grasp_metrics/{key.replace('grasp_metrics_', '')}", _finite_or_zero(values[i]), step)
        if include_joints and joint_pos is not None and joint_action is not None:
            for idx in range(joint_pos.shape[1]):
                writer.add_scalar(f"joint_pos/joint_{idx}", _finite_or_zero(joint_pos[i, idx]), step)
            for idx in range(joint_action.shape[1]):
                writer.add_scalar(f"joint_action/joint_{idx}", _finite_or_zero(joint_action[i, idx]), step)
        if include_observations and obs_state is not None:
            for idx in range(obs_state.shape[1]):
                writer.add_scalar(f"obs_state/dim_{idx:02d}", _finite_or_zero(obs_state[i, idx]), step)


def _write_dataset_summary(
    writer,
    dataset_summary: dict[str, Any],
    episode_summaries: list[dict[str, Any]],
    stage_maps: dict[str, dict[str, int]],
) -> None:
    writer.add_scalar("dataset/num_episodes", dataset_summary["num_episodes"], 0)
    writer.add_scalar("dataset/success_rate", dataset_summary["success_rate"], 0)
    writer.add_scalar("dataset/quality_rate", dataset_summary["quality_rate"], 0)
    writer.add_scalar(
        "dataset/mean_final_cube_place_dist_m",
        _finite_or_zero(dataset_summary["mean_final_cube_place_dist_m"]),
        0,
    )
    writer.add_text(
        "dataset/overview",
        _dataset_markdown(dataset_summary, episode_summaries, stage_maps),
        0,
    )


def _dataset_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(summaries)
    final_dists = [
        float(item["final_cube_place_dist_m"])
        for item in summaries
        if np.isfinite(float(item["final_cube_place_dist_m"]))
    ]
    return {
        "num_episodes": count,
        "success_count": sum(1 for item in summaries if item["success"]),
        "quality_ok_count": sum(1 for item in summaries if item["quality_ok"]),
        "success_rate": _safe_ratio(sum(1 for item in summaries if item["success"]), count),
        "quality_rate": _safe_ratio(sum(1 for item in summaries if item["quality_ok"]), count),
        "mean_final_cube_place_dist_m": float(np.mean(final_dists)) if final_dists else float("nan"),
    }


def _collect_stage_maps(episode_paths: list[Path]) -> dict[str, dict[str, int]]:
    labels: dict[str, set[str]] = {
        "phase": set(),
        "oriented_stage": set(),
        "physical_carry_stage": set(),
    }
    for path in episode_paths:
        data = np.load(path, allow_pickle=False)
        length = _length(data)
        for key in labels:
            labels[key].update(str(value) for value in _string_series(data, key, length) if value)
    return {
        key: {label: idx for idx, label in enumerate(sorted(values), start=1)}
        for key, values in labels.items()
    }


def _metadata_markdown(summary: dict[str, Any], stage_maps: dict[str, dict[str, int]]) -> str:
    warnings = summary.get("quality_warnings") or []
    lines = [
        f"# {summary['episode']}",
        "",
        f"- success: `{summary['success']}`",
        f"- quality_ok: `{summary['quality_ok']}`",
        f"- length: `{summary['length']}` steps",
        f"- duration: `{summary['duration_s']:.3f}` s",
        f"- final cube-place distance: `{summary['final_cube_place_dist_m']:.4f}` m",
        f"- active hand/cube: `{summary['active_hand']}` / `{summary['active_cube']}`",
        f"- instruction: {summary['instruction']}",
        f"- failure_reason: `{summary['failure_reason']}`",
        "",
        "## Stage IDs",
        _mapping_markdown("phase", stage_maps["phase"]),
        _mapping_markdown("oriented_stage", stage_maps["oriented_stage"]),
        _mapping_markdown("physical_carry_stage", stage_maps["physical_carry_stage"]),
    ]
    if warnings:
        lines += ["", "## Quality Warnings", *[f"- {warning}" for warning in warnings]]
    return "\n".join(lines)


def _dataset_markdown(
    dataset_summary: dict[str, Any],
    episode_summaries: list[dict[str, Any]],
    stage_maps: dict[str, dict[str, int]],
) -> str:
    lines = [
        "# Handover Dataset TensorBoard Export",
        "",
        f"- episodes: `{dataset_summary['num_episodes']}`",
        f"- success: `{dataset_summary['success_count']}`",
        f"- quality_ok: `{dataset_summary['quality_ok_count']}`",
        f"- success_rate: `{dataset_summary['success_rate']:.3f}`",
        f"- quality_rate: `{dataset_summary['quality_rate']:.3f}`",
        f"- mean final cube-place distance: `{dataset_summary['mean_final_cube_place_dist_m']:.4f}` m",
        "",
        "## Stage IDs",
        _mapping_markdown("phase", stage_maps["phase"]),
        _mapping_markdown("oriented_stage", stage_maps["oriented_stage"]),
        _mapping_markdown("physical_carry_stage", stage_maps["physical_carry_stage"]),
        "",
        "## Episodes",
        "| episode | success | quality_ok | length | final_dist_m | warning_count |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in episode_summaries:
        lines.append(
            f"| {item['episode']} | {item['success']} | {item['quality_ok']} | "
            f"{item['length']} | {item['final_cube_place_dist_m']:.4f} | "
            f"{len(item.get('quality_warnings') or [])} |"
        )
    return "\n".join(lines)


def _mapping_markdown(name: str, mapping: dict[str, int]) -> str:
    if not mapping:
        return f"- {name}: no labels"
    parts = ", ".join(f"`{idx}`={label}" for label, idx in mapping.items())
    return f"- {name}: {parts}"


def _write_vec3(writer, tag_prefix: str, values: np.ndarray, step: int) -> None:
    for axis, value in zip(("x", "y", "z"), np.asarray(values).reshape(-1)[:3]):
        writer.add_scalar(f"{tag_prefix}_{axis}", _finite_or_zero(value), step)


def _length(data) -> int:
    if "action" in data.files:
        return int(np.asarray(data["action"]).shape[0])
    if "obs_state" in data.files:
        return int(np.asarray(data["obs_state"]).shape[0])
    return 0


def _array(data, key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key in data.files:
        return np.asarray(data[key])
    return np.full(shape, np.nan, dtype=np.float32)


def _series(data, key: str, length: int) -> np.ndarray:
    if key in data.files:
        return np.asarray(data[key]).reshape(-1)
    return np.full(length, np.nan, dtype=np.float32)


def _bool_series(data, key: str, length: int) -> np.ndarray:
    if key in data.files:
        return np.asarray(data[key], dtype=bool).reshape(-1)
    return np.zeros(length, dtype=bool)


def _string_series(data, key: str, length: int) -> np.ndarray:
    if key not in data.files:
        return np.full(length, "", dtype=str)
    values = np.asarray(data[key]).astype(str).reshape(-1)
    if values.size >= length:
        return values[:length]
    padded = np.full(length, "", dtype=str)
    padded[: values.size] = values
    return padded


def _scalar_str(data, key: str) -> str:
    if key not in data.files:
        return ""
    arr = np.asarray(data[key])
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 0:
        return ""
    return str(arr.reshape(-1)[0])


def _scalar_bool(data, key: str) -> bool:
    if key not in data.files:
        return False
    try:
        return bool(np.asarray(data[key]).item())
    except ValueError:
        return False


def _first_nonempty(data, key: str) -> str:
    if key not in data.files:
        return ""
    for value in np.asarray(data[key]).reshape(-1):
        text = str(value)
        if text:
            return text
    return ""


def _value_counts(data, key: str) -> dict[str, int]:
    if key not in data.files:
        return {}
    values, counts = np.unique(np.asarray(data[key]).astype(str), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts) if str(value)}


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return _nan_stats()
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _nan_stats() -> dict[str, float]:
    return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}


def _final_dist(cube_pos: np.ndarray, place_pos: np.ndarray) -> float:
    if cube_pos.size == 0 or place_pos.size == 0:
        return float("nan")
    cube = np.asarray(cube_pos[-1], dtype=float)
    place = np.asarray(place_pos[-1], dtype=float)
    if not np.all(np.isfinite(cube)) or not np.all(np.isfinite(place)):
        return float("nan")
    return float(np.linalg.norm(cube - place))


def _first_true_index(values: np.ndarray) -> int | None:
    indices = np.where(np.asarray(values, dtype=bool))[0]
    if indices.size == 0:
        return None
    return int(indices[0])


def _label_id(mapping: dict[str, int], label: str) -> int:
    if not label:
        return 0
    return int(mapping.get(label, 0))


def _finite_or_zero(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _norm_or_zero(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        return 0.0
    return float(np.linalg.norm(arr))


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _safe_tag(value: str) -> str:
    return str(value).strip().replace("/", "_") or "empty"


if __name__ == "__main__":
    main()
