from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET_DIR = Path("data/trajectories/handover_dataset_v0")
METRIC_KEYS = (
    "grasp_metrics_center_dist",
    "grasp_metrics_min_dist",
    "grasp_metrics_width",
    "grasp_metrics_jaw_offset",
    "grasp_metrics_vertical_offset",
    "grasp_metrics_depth_offset",
    "grasp_metrics_head_dist",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and visualize handover expert dataset episodes."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_DATASET_DIR),
        help="Dataset directory or one episode_*.npz file.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for summaries, CSV files, and plots.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plots.")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV export.")
    parser.add_argument(
        "--success-dist",
        type=float,
        default=0.08,
        help="Maximum final cube-place distance considered a valid success.",
    )
    args = parser.parse_args()

    input_path = Path(args.path).expanduser()
    episodes = _episode_paths(input_path)
    if not episodes:
        raise SystemExit(f"No episode_*.npz files found at {input_path}")

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else _default_out_dir(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for episode_path in episodes:
        summary = inspect_episode(
            episode_path,
            out_dir=out_dir,
            write_csv=not args.no_csv,
            write_plots=not args.no_plots,
            success_dist_m=float(args.success_dist),
        )
        summaries.append(summary)
        _print_summary(summary)

    summary_path = out_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Inspect] wrote summary: {summary_path}")


def inspect_episode(
    episode_path: Path,
    *,
    out_dir: Path,
    write_csv: bool,
    write_plots: bool,
    success_dist_m: float,
) -> dict[str, Any]:
    data = np.load(episode_path, allow_pickle=False)
    length = _length(data)
    cube_pos = _array(data, "cube_pos", (length, 3))
    place_pos = _array(data, "place_pos", (length, 3))
    target_ee_pos = _array(data, "target_ee_pos", (length, 3))
    action = _array(data, "action", (length, 8))
    sim_time = _series(data, "sim_time", length)
    step = _series(data, "step", length).astype(np.int64)
    final_dist = _final_dist(cube_pos, place_pos)
    robot_has_taken_over = _bool_series(data, "robot_has_taken_over", length)
    takeover_idx = _first_true_index(robot_has_taken_over)
    success = bool(np.asarray(data["success"]).item()) if "success" in data.files else False
    failure_reason = _scalar_str(data, "failure_reason")
    instruction = _scalar_str(data, "instruction")
    quality_warnings = []
    if success and np.isfinite(final_dist) and final_dist > success_dist_m:
        quality_warnings.append(
            f"success=True but final cube-place dist {final_dist:.4f} m > {success_dist_m:.4f} m"
        )
    if not np.any(action[:, 7] < -0.5):
        quality_warnings.append("no close gripper command found")
    if not np.any(action[:, 7] > 0.5):
        quality_warnings.append("no open gripper command found")

    summary = {
        "episode": episode_path.name,
        "path": str(episode_path),
        "length": int(length),
        "success": success,
        "failure_reason": failure_reason,
        "instruction": instruction,
        "active_cube": _first_nonempty(data, "active_cube"),
        "active_hand": _first_nonempty(data, "active_hand"),
        "duration_s": float(sim_time[-1] - sim_time[0]) if length else 0.0,
        "final_cube_place_dist_m": final_dist,
        "quality_ok": len(quality_warnings) == 0,
        "quality_warnings": quality_warnings,
        "takeover_step_index": takeover_idx,
        "phase_counts": _value_counts(data, "phase"),
        "oriented_stage_counts": _value_counts(data, "oriented_stage"),
        "physical_carry_stage_counts": _value_counts(data, "physical_carry_stage"),
        "action_delta_stats": {
            axis: _finite_stats(action[:, idx])
            for idx, axis in enumerate(("dx", "dy", "dz"))
        },
        "gripper_command_counts": _gripper_counts(action[:, 7]),
    }

    stem = episode_path.stem
    summary_path = out_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)

    if write_csv:
        csv_path = out_dir / f"{stem}_timeseries.csv"
        _write_csv(data, csv_path)
        summary["csv_path"] = str(csv_path)

    if write_plots:
        plot_path = out_dir / f"{stem}_report.png"
        _write_plot(data, plot_path)
        summary["plot_path"] = str(plot_path)

    return summary


def _episode_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("episode_*.npz"))


def _default_out_dir(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.parent / "inspect"
    return input_path / "inspect"


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
    return np.arange(length, dtype=np.float32)


def _bool_series(data, key: str, length: int) -> np.ndarray:
    if key in data.files:
        return np.asarray(data[key], dtype=bool).reshape(-1)
    return np.zeros(length, dtype=bool)


def _scalar_str(data, key: str) -> str:
    if key not in data.files:
        return ""
    arr = np.asarray(data[key])
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 0:
        return ""
    return str(arr.reshape(-1)[0])


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
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _gripper_counts(values: np.ndarray) -> dict[str, int]:
    values = np.asarray(values, dtype=float)
    return {
        "open_or_positive": int(np.sum(values > 0.5)),
        "close_or_negative": int(np.sum(values < -0.5)),
        "neutral": int(np.sum(np.abs(values) <= 0.5)),
    }


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


def _write_csv(data, csv_path: Path) -> None:
    length = _length(data)
    cube_pos = _array(data, "cube_pos", (length, 3))
    place_pos = _array(data, "place_pos", (length, 3))
    target_ee_pos = _array(data, "target_ee_pos", (length, 3))
    action = _array(data, "action", (length, 8))
    joint_pos = _array(data, "joint_pos", (length, 9))
    joint_action = _array(data, "joint_action", (length, 9))
    step = _series(data, "step", length)
    sim_time = _series(data, "sim_time", length)
    phase = _string_series(data, "phase", length)
    oriented_stage = _string_series(data, "oriented_stage", length)
    carry_stage = _string_series(data, "physical_carry_stage", length)
    active_hand = _string_series(data, "active_hand", length)
    active_cube = _string_series(data, "active_cube", length)
    robot_has_taken_over = _bool_series(data, "robot_has_taken_over", length)

    headers = [
        "i",
        "step",
        "sim_time",
        "cube_x",
        "cube_y",
        "cube_z",
        "place_x",
        "place_y",
        "place_z",
        "target_ee_x",
        "target_ee_y",
        "target_ee_z",
        "action_dx",
        "action_dy",
        "action_dz",
        "action_qw",
        "action_qx",
        "action_qy",
        "action_qz",
        "action_gripper",
    ]
    headers += [f"joint_pos_{idx}" for idx in range(9)]
    headers += [f"joint_action_{idx}" for idx in range(9)]
    headers += METRIC_KEYS
    headers += [
        "phase",
        "oriented_stage",
        "physical_carry_stage",
        "active_hand",
        "active_cube",
        "robot_has_taken_over",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(length):
            row: list[Any] = [
                i,
                int(step[i]) if np.isfinite(step[i]) else "",
                float(sim_time[i]) if np.isfinite(sim_time[i]) else "",
                *[float(v) for v in cube_pos[i]],
                *[float(v) for v in place_pos[i]],
                *[float(v) for v in target_ee_pos[i]],
                *[float(v) for v in action[i]],
                *[float(v) for v in joint_pos[i]],
                *[float(v) for v in joint_action[i]],
            ]
            for key in METRIC_KEYS:
                values = _series(data, key, length)
                row.append(float(values[i]) if i < values.size and np.isfinite(values[i]) else "")
            row += [
                phase[i],
                oriented_stage[i],
                carry_stage[i],
                active_hand[i],
                active_cube[i],
                int(robot_has_taken_over[i]),
            ]
            writer.writerow(row)


def _string_series(data, key: str, length: int) -> np.ndarray:
    if key not in data.files:
        return np.full(length, "", dtype=str)
    values = np.asarray(data[key]).astype(str).reshape(-1)
    if values.size >= length:
        return values[:length]
    padded = np.full(length, "", dtype=str)
    padded[: values.size] = values
    return padded


def _write_plot(data, plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    length = _length(data)
    idx = np.arange(length)
    sim_time = _series(data, "sim_time", length)
    x = sim_time - sim_time[0] if length else idx
    cube_pos = _array(data, "cube_pos", (length, 3))
    place_pos = _array(data, "place_pos", (length, 3))
    target_ee_pos = _array(data, "target_ee_pos", (length, 3))
    action = _array(data, "action", (length, 8))
    robot_has_taken_over = _bool_series(data, "robot_has_taken_over", length)

    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle(
        f"{plot_path.stem.replace('_report', '')} | "
        f"success={_scalar_str(data, 'success')} | "
        f"{_scalar_str(data, 'instruction')}",
        fontsize=12,
    )

    ax = axes[0, 0]
    ax.plot(cube_pos[:, 0], cube_pos[:, 1], label="cube", color="#d95f02")
    ax.plot(target_ee_pos[:, 0], target_ee_pos[:, 1], label="target EE", color="#1b9e77")
    ax.scatter(place_pos[:, 0], place_pos[:, 1], label="place", color="#7570b3", s=8)
    ax.scatter(cube_pos[0, 0], cube_pos[0, 1], marker="o", color="#d95f02", s=60, label="cube start")
    ax.scatter(cube_pos[-1, 0], cube_pos[-1, 1], marker="x", color="#d95f02", s=80, label="cube end")
    ax.set_title("Top-Down XY Trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, cube_pos[:, 2], label="cube z", color="#d95f02")
    ax.plot(x, target_ee_pos[:, 2], label="target EE z", color="#1b9e77")
    ax.plot(x, place_pos[:, 2], label="place z", color="#7570b3")
    ax.set_title("Height Over Time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("z [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(x, action[:, 0], label="dx")
    ax.plot(x, action[:, 1], label="dy")
    ax.plot(x, action[:, 2], label="dz")
    ax.set_title("Action Delta: target EE - current EE")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("delta [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(x, action[:, 7], label="gripper command", color="#e7298a")
    ax.set_title("Gripper Command: open=1, close=-1")
    ax.set_xlabel("time [s]")
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    for key in METRIC_KEYS:
        if key in data.files:
            label = key.replace("grasp_metrics_", "")
            ax.plot(x, np.asarray(data[key]), label=label)
    ax.set_title("Grasp Metrics")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance / width [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2, 1]
    stages = _string_series(data, "oriented_stage", length)
    carry = _string_series(data, "physical_carry_stage", length)
    labels = list(dict.fromkeys([s for s in stages if s] + [f"carry:{s}" for s in carry if s]))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    stage_y = np.full(length, np.nan)
    for i, stage in enumerate(stages):
        if stage:
            stage_y[i] = label_to_idx[stage]
    carry_y = np.full(length, np.nan)
    for i, stage in enumerate(carry):
        if stage:
            carry_y[i] = label_to_idx[f"carry:{stage}"]
    ax.scatter(x, stage_y, s=6, label="oriented stage")
    ax.scatter(x, carry_y, s=6, label="carry stage")
    if np.any(robot_has_taken_over):
        takeover = _first_true_index(robot_has_taken_over)
        if takeover is not None:
            ax.axvline(x[takeover], color="black", linestyle="--", linewidth=1, label="takeover")
    ax.set_title("FSM Stages")
    ax.set_xlabel("time [s]")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"\n[Inspect] {summary['episode']}")
    print(f"  success: {summary['success']} failure_reason={summary['failure_reason']!r}")
    print(f"  length: {summary['length']} duration_s={summary['duration_s']:.2f}")
    print(f"  active: hand={summary['active_hand']} cube={summary['active_cube']}")
    print(f"  final cube-place dist: {summary['final_cube_place_dist_m']:.4f} m")
    print(f"  quality_ok: {summary['quality_ok']}")
    for warning in summary.get("quality_warnings", []):
        print(f"    WARNING: {warning}")
    print(f"  takeover step index: {summary['takeover_step_index']}")
    print(f"  gripper counts: {summary['gripper_command_counts']}")
    print(f"  physical carry counts: {summary['physical_carry_stage_counts']}")
    if "plot_path" in summary:
        print(f"  plot: {summary['plot_path']}")
    if "csv_path" in summary:
        print(f"  csv: {summary['csv_path']}")


if __name__ == "__main__":
    main()
