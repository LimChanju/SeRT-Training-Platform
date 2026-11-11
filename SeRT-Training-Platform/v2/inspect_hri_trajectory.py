from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_INPUT = Path(__file__).resolve().parent / "trajectories" / "hri_vr_expert_v0.hdf5"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect HRI VR trajectory files for coordinate and risk-label sanity."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="HDF5 HRI trajectory path.")
    parser.add_argument("--output-json", default="", help="Optional JSON summary output path.")
    parser.add_argument("--output-md", default="", help="Optional Markdown summary output path.")
    parser.add_argument("--near-dist", type=float, default=0.12, help="Near-human threshold in meters.")
    parser.add_argument(
        "--collision-dist",
        type=float,
        default=0.05,
        help="Collision-candidate distance in meters.",
    )
    parser.add_argument(
        "--workspace-margin",
        type=float,
        default=0.20,
        help="Warning margin around cube/place-target xy extents.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = Path(args.input).expanduser().resolve()
    summary = inspect_hri_trajectory(
        path,
        near_dist=float(args.near_dist),
        collision_dist=float(args.collision_dist),
        workspace_margin=float(args.workspace_margin),
    )
    print(_format_console(summary))
    if args.output_json:
        output = Path(args.output_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[InspectHRI] saved json={output}")
    if args.output_md:
        output = Path(args.output_md).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_format_markdown(summary), encoding="utf-8")
        print(f"[InspectHRI] saved md={output}")


def inspect_hri_trajectory(
    path: Path,
    *,
    near_dist: float,
    collision_dist: float,
    workspace_margin: float,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as h5:
        if "episodes" not in h5:
            raise KeyError(f"Missing /episodes group: {path}")
        episode_names = sorted(h5["episodes"].keys())
        episodes = [
            _inspect_episode(
                h5["episodes"][name],
                name=name,
                near_dist=near_dist,
                collision_dist=collision_dist,
                workspace_margin=workspace_margin,
            )
            for name in episode_names
        ]
        warnings = []
        for episode in episodes:
            warnings.extend(f"{episode['name']}: {warning}" for warning in episode["warnings"])
        root_attrs = _attrs_dict(h5.attrs)
    return {
        "path": str(path),
        "file_size_bytes": int(path.stat().st_size),
        "root_attrs": root_attrs,
        "episode_count": len(episodes),
        "episodes": episodes,
        "warnings": warnings,
    }


def _inspect_episode(
    group,
    *,
    name: str,
    near_dist: float,
    collision_dist: float,
    workspace_margin: float,
) -> dict[str, Any]:
    attrs = _attrs_dict(group.attrs)
    obs = group["obs"] if "obs" in group else None
    human = group["human"] if "human" in group else None
    length = int(attrs.get("episode_length", _infer_length(group)))

    head = _dataset(human, "head_pos", length, 3)
    left = _dataset(human, "left_hand_pos", length, 3)
    right = _dataset(human, "right_hand_pos", length, 3)
    left_vel = _dataset(human, "left_hand_vel", length, 3)
    right_vel = _dataset(human, "right_hand_vel", length, 3)
    valid = _dataset(human, "valid_mask", length, 3)
    if not np.any(valid):
        valid = _derived_valid_mask(head, left, right)

    ee = _dataset(obs, "ee_pos", length, 3)
    cube = _dataset(obs, "cube_pos", length, 3)
    target = _dataset(obs, "place_target_pos", length, 3)
    recorded_min = _dataset(obs, "min_hand_gripper_dist", length, 1).reshape(-1)
    recorded_near = _dataset(obs, "near_human", length, 1).reshape(-1)
    recorded_collision = _dataset(obs, "human_robot_collision", length, 1).reshape(-1)

    computed_left = _distance_series(ee, left, valid[:, 1])
    computed_right = _distance_series(ee, right, valid[:, 2])
    computed_min = np.minimum(computed_left, computed_right)
    computed_min[~np.isfinite(computed_min)] = np.nan
    computed_near = computed_min < near_dist
    computed_collision = computed_min < collision_dist

    workspace = _workspace_bounds(cube, target, margin=workspace_margin)
    left_inside = _inside_xy_fraction(left, valid[:, 1], workspace)
    right_inside = _inside_xy_fraction(right, valid[:, 2], workspace)

    dist_error = np.abs(recorded_min - computed_min)
    finite_error = dist_error[np.isfinite(dist_error) & np.isfinite(recorded_min)]
    warnings = []
    if float(np.mean(valid[:, 1] > 0.5)) < 0.5 and float(np.mean(valid[:, 2] > 0.5)) < 0.5:
        warnings.append("few valid hand samples")
    if np.isfinite(computed_min).any() and float(np.nanmin(computed_min)) > near_dist:
        warnings.append("hands never enter near-human threshold")
    if finite_error.size and float(np.nanmedian(finite_error)) > 0.08:
        warnings.append("recorded min distance differs from ee-hand approximation; verify gripper frame")
    if left_inside < 0.2 and right_inside < 0.2:
        warnings.append("hands mostly outside cube/target workspace bounds; check rotation/offset")
    if _flag_rate(recorded_near) == 0.0 and float(np.mean(computed_near[np.isfinite(computed_min)])) > 0.05:
        warnings.append("computed near-human events exist but recorded near_human is always zero")
    if _flag_rate(recorded_collision) > 0.0 and not np.any(computed_collision):
        warnings.append("recorded collisions exist without close ee-hand distance; check collision geometry")

    return {
        "name": name,
        "attrs": attrs,
        "length": length,
        "success": bool(attrs.get("success", False)),
        "coordinate_frame": str(attrs.get("coordinate_frame", "")),
        "human_position_frame": str(attrs.get("human_position_frame", "")),
        "left_hand": _series_summary(left, valid[:, 1], left_vel),
        "right_hand": _series_summary(right, valid[:, 2], right_vel),
        "head": _series_summary(head, valid[:, 0], None),
        "workspace_xy_bounds": workspace,
        "left_inside_workspace_xy_fraction": left_inside,
        "right_inside_workspace_xy_fraction": right_inside,
        "recorded_min_hand_gripper_dist": _numeric_summary(recorded_min),
        "computed_min_ee_hand_dist": _numeric_summary(computed_min),
        "min_distance_abs_error_median": _nan_stat(finite_error, "median"),
        "recorded_near_human_rate": _flag_rate(recorded_near),
        "computed_near_human_rate": _flag_rate(computed_near),
        "recorded_collision_rate": _flag_rate(recorded_collision),
        "computed_collision_candidate_rate": _flag_rate(computed_collision),
        "warnings": warnings,
    }


def _dataset(group, name: str, length: int, width: int) -> np.ndarray:
    if group is not None and name in group:
        arr = np.asarray(group[name], dtype=np.float32)
        return arr.reshape((arr.shape[0], width))
    return np.zeros((length, width), dtype=np.float32)


def _infer_length(group) -> int:
    for child_name in ("sim_time", "obs_policy", "dones"):
        if child_name in group:
            return int(group[child_name].shape[0])
    if "obs" in group:
        for dataset in group["obs"].values():
            return int(dataset.shape[0])
    return 0


def _derived_valid_mask(head: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.stack([_valid(head), _valid(left), _valid(right)], axis=1).astype(np.float32)


def _valid(values: np.ndarray) -> np.ndarray:
    return np.logical_and(np.all(np.isfinite(values), axis=1), np.linalg.norm(values, axis=1) > 1e-6)


def _distance_series(origin: np.ndarray, points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(points - origin, axis=1).astype(np.float32)
    dist[valid <= 0.5] = np.nan
    return dist


def _workspace_bounds(cube: np.ndarray, target: np.ndarray, *, margin: float) -> dict[str, float]:
    xy = np.concatenate([cube[:, :2], target[:, :2]], axis=0)
    finite = np.all(np.isfinite(xy), axis=1)
    xy = xy[finite]
    if xy.size == 0:
        return {"x_min": -np.inf, "x_max": np.inf, "y_min": -np.inf, "y_max": np.inf}
    return {
        "x_min": float(np.min(xy[:, 0]) - margin),
        "x_max": float(np.max(xy[:, 0]) + margin),
        "y_min": float(np.min(xy[:, 1]) - margin),
        "y_max": float(np.max(xy[:, 1]) + margin),
    }


def _inside_xy_fraction(points: np.ndarray, valid: np.ndarray, bounds: dict[str, float]) -> float:
    mask = valid > 0.5
    if not np.any(mask):
        return 0.0
    xy = points[mask, :2]
    inside = (
        (xy[:, 0] >= bounds["x_min"])
        & (xy[:, 0] <= bounds["x_max"])
        & (xy[:, 1] >= bounds["y_min"])
        & (xy[:, 1] <= bounds["y_max"])
    )
    return float(np.mean(inside))


def _series_summary(pos: np.ndarray, valid: np.ndarray, vel: np.ndarray | None) -> dict[str, Any]:
    mask = valid > 0.5
    valid_pos = pos[mask]
    out = {
        "valid_fraction": float(np.mean(mask)) if len(mask) else 0.0,
        "pos_min": _vec_stat(valid_pos, "min"),
        "pos_max": _vec_stat(valid_pos, "max"),
        "pos_mean": _vec_stat(valid_pos, "mean"),
    }
    if vel is not None:
        speed = np.linalg.norm(vel[mask], axis=1) if np.any(mask) else np.asarray([], dtype=np.float32)
        out["speed_mps"] = _numeric_summary(speed)
    return out


def _numeric_summary(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def _vec_stat(values: np.ndarray, stat: str) -> list[float | None]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return [None, None, None]
    if stat == "min":
        return [float(x) for x in np.min(values, axis=0)]
    if stat == "max":
        return [float(x) for x in np.max(values, axis=0)]
    if stat == "mean":
        return [float(x) for x in np.mean(values, axis=0)]
    raise ValueError(stat)


def _nan_stat(values: np.ndarray, stat: str) -> float | None:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    if stat == "median":
        return float(np.median(values))
    raise ValueError(stat)


def _flag_rate(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.mean(values > 0.5))


def _attrs_dict(attrs) -> dict[str, Any]:
    out = {}
    for key, value in attrs.items():
        out[str(key)] = _jsonable(value)
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _format_console(summary: dict[str, Any]) -> str:
    lines = [
        f"[InspectHRI] path={summary['path']}",
        f"[InspectHRI] episodes={summary['episode_count']} warnings={len(summary['warnings'])}",
    ]
    for episode in summary["episodes"]:
        lines.append(
            "[InspectHRI] "
            f"{episode['name']} len={episode['length']} success={episode['success']} "
            f"min_dist={episode['computed_min_ee_hand_dist']['min']} "
            f"near={episode['recorded_near_human_rate']:.3f}/computed={episode['computed_near_human_rate']:.3f} "
            f"inside_xy L={episode['left_inside_workspace_xy_fraction']:.3f} "
            f"R={episode['right_inside_workspace_xy_fraction']:.3f}"
        )
        for warning in episode["warnings"]:
            lines.append(f"[InspectHRI][warning] {episode['name']}: {warning}")
    return "\n".join(lines)


def _format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# HRI Trajectory Inspection",
        "",
        f"- Path: `{summary['path']}`",
        f"- Episodes: {summary['episode_count']}",
        f"- Warnings: {len(summary['warnings'])}",
        "",
        "## Episodes",
        "",
    ]
    for episode in summary["episodes"]:
        min_dist = episode["computed_min_ee_hand_dist"]["min"]
        lines.extend(
            [
                f"### {episode['name']}",
                "",
                f"- Length: {episode['length']}",
                f"- Success: {episode['success']}",
                f"- Coordinate frame: `{episode['coordinate_frame']}`",
                f"- Human position frame: `{episode['human_position_frame']}`",
                f"- Computed min EE-hand distance: {min_dist}",
                f"- Recorded near-human rate: {episode['recorded_near_human_rate']:.4f}",
                f"- Computed near-human rate: {episode['computed_near_human_rate']:.4f}",
                f"- Left hand inside workspace XY: {episode['left_inside_workspace_xy_fraction']:.4f}",
                f"- Right hand inside workspace XY: {episode['right_inside_workspace_xy_fraction']:.4f}",
                "",
            ]
        )
        if episode["warnings"]:
            lines.append("Warnings:")
            for warning in episode["warnings"]:
                lines.append(f"- {warning}")
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
