from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v3_chan.rl.observations import (  # noqa: E402
    HRI_OBS_FIELD_NAMES,
    OBSERVATION_FIELDS,
    _FIELD_MAP,
)


PREPROCESSING_VERSION = "hri_training_ready_v1"
GRASP_DISTANCE_M = 0.075
GRASP_WIDTH_M = 0.065


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field_slices(fields) -> dict[str, slice]:
    cursor = 0
    slices: dict[str, slice] = {}
    for field in fields:
        slices[field.name] = slice(cursor, cursor + field.dim)
        cursor += field.dim
    return slices


def _hri_field_slices() -> dict[str, slice]:
    cursor = 0
    slices: dict[str, slice] = {}
    for name in HRI_OBS_FIELD_NAMES:
        field = _FIELD_MAP[name]
        slices[name] = slice(cursor, cursor + field.dim)
        cursor += field.dim
    return slices


OBS_SLICES = _field_slices(OBSERVATION_FIELDS)
HRI_SLICES = _hri_field_slices()


def _runs(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0:
        return 0
    return int(np.sum(values & ~np.r_[False, values[:-1]]))


def _expand_mask(mask: np.ndarray, padding: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if padding <= 0 or not np.any(values):
        return values.copy()
    kernel = np.ones(2 * padding + 1, dtype=np.uint8)
    return np.convolve(values.astype(np.uint8), kernel, mode="same") > 0


def _write_mask(group: h5py.Group, name: str, values: np.ndarray) -> None:
    group.create_dataset(
        name,
        data=np.asarray(values, dtype=np.uint8),
        compression="gzip",
        shuffle=True,
    )


def _validate_source(data: h5py.File, source: Path) -> list[str]:
    if "episodes" not in data:
        raise ValueError(f"{source}: missing episodes group")
    names = sorted(data["episodes"].keys())
    if len(names) != 3:
        raise ValueError(f"{source}: expected 3 episodes, found {len(names)}")
    for name in names:
        episode = data["episodes"][name]
        if not bool(episode.attrs.get("success", False)):
            raise ValueError(f"{source}: {name} is not successful")
        if int(episode.attrs.get("failed_attempts", -1)) != 0:
            raise ValueError(f"{source}: {name} contains failed attempts")
        if episode["obs_policy"].shape[1] != 84:
            raise ValueError(f"{source}: {name} obs_policy is not 84-D")
        if episode["hri_obs_policy"].shape[1] != 83:
            raise ValueError(f"{source}: {name} hri_obs_policy is not 83-D")
    return names


def _process_episode(
    episode: h5py.Group,
    *,
    episode_index: int,
    warmup_frames: int,
    velocity_threshold_mps: float,
    velocity_padding_frames: int,
) -> dict:
    length = int(episode.attrs["episode_length"])
    cube_pos = np.asarray(episode["obs/cube_pos"][...], dtype=np.float64)
    ee_pos = np.asarray(episode["obs/ee_pos"][...], dtype=np.float64)
    gripper_width = np.asarray(
        episode["obs/gripper_width"][...], dtype=np.float64
    ).reshape(-1)
    if cube_pos.shape != (length, 3) or ee_pos.shape != (length, 3):
        raise ValueError(f"{episode.name}: unexpected cube or end-effector shape")

    grasp_distance = np.linalg.norm(cube_pos - ee_pos, axis=1)
    grasp = (grasp_distance < GRASP_DISTANCE_M) & (
        gripper_width < GRASP_WIDTH_M
    )
    source_grasp = np.asarray(
        episode["task/has_grasped_cube"][...], dtype=np.float32
    ).reshape(-1)

    preprocessing = episode.create_group("preprocessing")
    _write_mask(preprocessing, "source_has_grasped_cube", source_grasp > 0.5)
    _write_mask(preprocessing, "has_grasped_cube_reconstructed", grasp)

    episode["obs/has_grasped_cube"][...] = grasp.astype(np.float32).reshape(-1, 1)
    episode["hri_obs/has_grasped_cube"][...] = grasp.astype(np.float32).reshape(-1, 1)
    episode["task/has_grasped_cube"][...] = grasp.astype(
        episode["task/has_grasped_cube"].dtype
    )

    obs_slice = OBS_SLICES["has_grasped_cube"]
    hri_slice = HRI_SLICES["has_grasped_cube"]
    episode["obs_policy"][:, obs_slice] = grasp.astype(np.float32).reshape(-1, 1)
    episode["hri_obs_policy"][:, hri_slice] = grasp.astype(np.float32).reshape(-1, 1)

    warmup_excluded = np.zeros(length, dtype=bool)
    if episode_index == 0:
        warmup_excluded[: min(length, warmup_frames)] = True

    left_velocity = np.asarray(
        episode["human/left_hand_vel_filtered_mps"][...], dtype=np.float64
    )
    right_velocity = np.asarray(
        episode["human/right_hand_vel_filtered_mps"][...], dtype=np.float64
    )
    left_speed = np.linalg.norm(left_velocity, axis=1)
    right_speed = np.linalg.norm(right_velocity, axis=1)
    left_velocity_core = left_speed > velocity_threshold_mps
    right_velocity_core = right_speed > velocity_threshold_mps
    velocity_core = left_velocity_core | right_velocity_core
    velocity_excluded = _expand_mask(velocity_core, velocity_padding_frames)

    tracking_valid = (
        np.asarray(episode["human/left_hand_pose_valid"][...], dtype=bool)
        & np.asarray(episode["human/right_hand_pose_valid"][...], dtype=bool)
    )
    dynamic_valid = np.asarray(
        episode["safety/dynamic_measurement_valid"][...], dtype=bool
    )
    rtf_valid = np.asarray(episode["real_time_factor_valid"][...], dtype=bool)
    training_valid = (
        tracking_valid
        & dynamic_valid
        & rtf_valid
        & ~warmup_excluded
        & ~velocity_excluded
    )

    _write_mask(preprocessing, "training_valid_mask", training_valid)
    _write_mask(preprocessing, "tracking_valid_mask", tracking_valid)
    _write_mask(preprocessing, "dynamic_measurement_valid_mask", dynamic_valid)
    _write_mask(preprocessing, "real_time_factor_valid_mask", rtf_valid)
    _write_mask(preprocessing, "anchor_warmup_exclusion_mask", warmup_excluded)
    _write_mask(preprocessing, "velocity_outlier_core_mask", velocity_core)
    _write_mask(preprocessing, "velocity_outlier_exclusion_mask", velocity_excluded)
    _write_mask(preprocessing, "left_velocity_outlier_core_mask", left_velocity_core)
    _write_mask(preprocessing, "right_velocity_outlier_core_mask", right_velocity_core)

    episode.attrs["preprocessing_version"] = PREPROCESSING_VERSION
    episode.attrs["preprocessing_training_valid_frames"] = int(training_valid.sum())
    episode.attrs["preprocessing_excluded_frames"] = int((~training_valid).sum())
    episode.attrs["preprocessing_reconstructed_grasp_frames"] = int(grasp.sum())
    episode.attrs["preprocessing_reconstructed_grasp_runs"] = _runs(grasp)

    outliers = []
    profile = str(episode.attrs.get("controller_speed_profile", ""))
    for side, core, speed in (
        ("left", left_velocity_core, left_speed),
        ("right", right_velocity_core, right_speed),
    ):
        for frame in np.flatnonzero(core):
            outliers.append(
                {
                    "side": side,
                    "frame": int(frame),
                    "speed_profile": profile,
                    "filtered_speed_mps": float(speed[frame]),
                }
            )

    return {
        "episode": episode.name.rsplit("/", 1)[-1],
        "speed_profile": profile,
        "frames": length,
        "training_valid_frames": int(training_valid.sum()),
        "excluded_frames": int((~training_valid).sum()),
        "warmup_excluded_frames": int(warmup_excluded.sum()),
        "tracking_invalid_frames": int((~tracking_valid).sum()),
        "dynamic_invalid_frames": int((~dynamic_valid).sum()),
        "rtf_invalid_frames": int((~rtf_valid).sum()),
        "velocity_outlier_core_frames": int(velocity_core.sum()),
        "velocity_outlier_excluded_frames": int(velocity_excluded.sum()),
        "reconstructed_grasp_frames": int(grasp.sum()),
        "reconstructed_grasp_runs": _runs(grasp),
        "source_grasp_frames": int((source_grasp > 0.5).sum()),
        "velocity_outliers": outliers,
    }


def _process_file(
    source: Path,
    output_dir: Path,
    *,
    warmup_frames: int,
    velocity_threshold_mps: float,
    velocity_padding_frames: int,
) -> dict:
    source_sha256 = _sha256(source)
    destination = output_dir / f"{source.stem}_{PREPROCESSING_VERSION}.hdf5"
    shutil.copy2(source, destination)

    with h5py.File(destination, "r+") as data:
        episode_names = _validate_source(data, source)
        summaries = []
        for index, name in enumerate(episode_names):
            summaries.append(
                _process_episode(
                    data["episodes"][name],
                    episode_index=index,
                    warmup_frames=warmup_frames,
                    velocity_threshold_mps=velocity_threshold_mps,
                    velocity_padding_frames=velocity_padding_frames,
                )
            )

        data.attrs["preprocessing_version"] = PREPROCESSING_VERSION
        data.attrs["preprocessing_source_filename"] = source.name
        data.attrs["preprocessing_source_sha256"] = source_sha256
        data.attrs["preprocessing_created_at_utc"] = datetime.now(timezone.utc).isoformat()
        data.attrs["preprocessing_frame_alignment_preserved"] = 1
        data.attrs["preprocessing_frames_physically_removed"] = 0
        data.attrs["preprocessing_grasp_method"] = (
            "norm(obs/cube_pos-obs/ee_pos)<0.075 and obs/gripper_width<0.065"
        )
        data.attrs["preprocessing_grasp_distance_m"] = GRASP_DISTANCE_M
        data.attrs["preprocessing_grasp_width_m"] = GRASP_WIDTH_M
        data.attrs["preprocessing_obs_policy_grasp_index"] = int(
            OBS_SLICES["has_grasped_cube"].start
        )
        data.attrs["preprocessing_hri_obs_policy_grasp_index"] = int(
            HRI_SLICES["has_grasped_cube"].start
        )
        data.attrs["preprocessing_first_episode_warmup_frames"] = warmup_frames
        data.attrs["preprocessing_velocity_outlier_threshold_mps"] = (
            velocity_threshold_mps
        )
        data.attrs["preprocessing_velocity_outlier_padding_frames"] = (
            velocity_padding_frames
        )
        data.attrs["preprocessing_training_valid_mask_semantics"] = (
            "both hands tracked AND dynamic_measurement_valid AND "
            "real_time_factor_valid AND not first-episode anchor warmup AND "
            "not expanded filtered-hand-speed outlier"
        )
        data.flush()

        session_id = str(data.attrs["session_id"])
        session_index = int(data.attrs["participant_session_index"])
        layout_id = str(data.attrs["session_layout_id"])
        speed_order = str(data.attrs["controller_speed_schedule"])

    return {
        "session_id": session_id,
        "participant_session_index": session_index,
        "layout_id": layout_id,
        "speed_order": speed_order,
        "source_file": source.name,
        "source_sha256": source_sha256,
        "processed_file": destination.name,
        "processed_sha256": _sha256(destination),
        "episodes": summaries,
        "frames": int(sum(item["frames"] for item in summaries)),
        "training_valid_frames": int(
            sum(item["training_valid_frames"] for item in summaries)
        ),
        "excluded_frames": int(sum(item["excluded_frames"] for item in summaries)),
        "reconstructed_grasp_frames": int(
            sum(item["reconstructed_grasp_frames"] for item in summaries)
        ),
        "reconstructed_grasp_runs": int(
            sum(item["reconstructed_grasp_runs"] for item in summaries)
        ),
    }


def _copy_logs(logs_dir: Path, output_dir: Path, session_ids: list[str]) -> list[dict]:
    copied = []
    for session_id in session_ids:
        matches = sorted(path for path in logs_dir.glob(f"*{session_id}*") if path.is_file())
        if len(matches) != 3:
            raise ValueError(
                f"session {session_id}: expected 3 companion log files, found {len(matches)}"
            )
        for source in matches:
            destination = output_dir / source.name
            shutil.copy2(source, destination)
            copied.append(
                {
                    "session_id": session_id,
                    "file": destination.name,
                    "sha256": _sha256(destination),
                }
            )
    return copied


def _write_readme(path: Path, manifest: dict) -> None:
    sessions = manifest["sessions"]
    total_frames = sum(item["frames"] for item in sessions)
    valid_frames = sum(item["training_valid_frames"] for item in sessions)
    lines = [
        "# P01 VR-HRI sessions 17-24: training-ready v1",
        "",
        "This package contains eight successful production sessions collected from 2026-08-04 to 2026-08-06.",
        "Each session contains three successful episodes using slow, medium, and fast robot speed profiles.",
        "",
        "## Contents",
        "",
        "- `hdf5/`: eight processed HDF5 files",
        "- `logs/`: Isaac logs, frame CSVs, and ErrP marker CSVs",
        "- `tools/preprocess_hri_release.py`: preprocessing implementation used for this package",
        "- `manifest.json`: source and processed checksums plus per-episode QC",
        "- `SHA256SUMS`: package checksums",
        "",
        "## Preprocessing",
        "",
        "- Original files were copied; no source HDF5 was modified.",
        "- No rows were removed, so temporal and EEG/event alignment remains intact.",
        "- `has_grasped_cube` was reconstructed from stored cube/EE distance and gripper width.",
        "- Canonical `obs`, `hri_obs`, `task`, `obs_policy`, and `hri_obs_policy` grasp values were updated consistently.",
        "- Use `episodes/*/preprocessing/training_valid_mask == 1` for dynamic-safety training.",
        "- The mask excludes first-episode anchor warmup, invalid tracking/dynamics/RTF, and isolated velocity spikes with padding.",
        "- TTC-invalid rows are retained because they correctly represent non-closing motion.",
        "",
        "## Summary",
        "",
        f"- Sessions: {len(sessions)}",
        f"- Episodes: {sum(len(item['episodes']) for item in sessions)}",
        f"- Frames: {total_frames}",
        f"- Training-valid frames: {valid_frames} ({100.0 * valid_frames / total_frames:.3f}%)",
        f"- Reconstructed grasp runs: {sum(item['reconstructed_grasp_runs'] for item in sessions)}",
        "",
        "Split cross-validation by `session_id` or `layout_id`; never split these files by random frames.",
        "Haptic pulse fields confirm successful UDP sends only, not physical glove vibration.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")


def _write_checksums(package_dir: Path) -> None:
    files = sorted(
        path
        for path in package_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{_sha256(path)}  {path.relative_to(package_dir)}" for path in files]
    (package_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create alignment-preserving, training-ready HRI release files."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--logs-dir", type=Path)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--velocity-threshold-mps", type=float, default=2.0)
    parser.add_argument("--velocity-padding-frames", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_dir = args.output_dir.resolve()
    if package_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output directory already exists: {package_dir}")
        shutil.rmtree(package_dir)
    hdf5_dir = package_dir / "hdf5"
    logs_output_dir = package_dir / "logs"
    hdf5_dir.mkdir(parents=True)
    logs_output_dir.mkdir()

    sources = sorted(path.resolve() for path in args.inputs)
    sessions = [
        _process_file(
            source,
            hdf5_dir,
            warmup_frames=args.warmup_frames,
            velocity_threshold_mps=args.velocity_threshold_mps,
            velocity_padding_frames=args.velocity_padding_frames,
        )
        for source in sources
    ]
    session_indices = [item["participant_session_index"] for item in sessions]
    if session_indices != list(range(17, 25)):
        raise ValueError(f"expected session indices 17..24, found {session_indices}")

    logs = []
    if args.logs_dir is not None:
        logs = _copy_logs(
            args.logs_dir.resolve(),
            logs_output_dir,
            [item["session_id"] for item in sessions],
        )

    tools_dir = package_dir / "tools"
    tools_dir.mkdir()
    tool_source = Path(__file__).resolve()
    tool_destination = tools_dir / tool_source.name
    shutil.copy2(tool_source, tool_destination)

    manifest = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "participant_id": "P01",
        "session_indices": session_indices,
        "source_date_range": "2026-08-04/2026-08-06",
        "frame_alignment_preserved": True,
        "frames_physically_removed": False,
        "preprocessing_tool": {
            "file": str(tool_destination.relative_to(package_dir)),
            "sha256": _sha256(tool_destination),
        },
        "grasp_reconstruction": {
            "distance_reference": "obs/cube_pos minus obs/ee_pos",
            "max_distance_m": GRASP_DISTANCE_M,
            "max_gripper_width_m": GRASP_WIDTH_M,
            "obs_policy_index": OBS_SLICES["has_grasped_cube"].start,
            "hri_obs_policy_index": HRI_SLICES["has_grasped_cube"].start,
        },
        "training_valid_mask": {
            "first_episode_warmup_frames": args.warmup_frames,
            "filtered_hand_speed_outlier_threshold_mps": args.velocity_threshold_mps,
            "velocity_outlier_padding_frames": args.velocity_padding_frames,
            "requires_both_hand_tracking_valid": True,
            "requires_dynamic_measurement_valid": True,
            "requires_real_time_factor_valid": True,
        },
        "sessions": sessions,
        "logs": logs,
    }
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    _write_readme(package_dir / "README.md", manifest)
    _write_checksums(package_dir)

    total_frames = sum(item["frames"] for item in sessions)
    valid_frames = sum(item["training_valid_frames"] for item in sessions)
    print(f"package={package_dir}")
    print(f"sessions={len(sessions)} frames={total_frames}")
    print(
        f"training_valid_frames={valid_frames} "
        f"fraction={valid_frames / total_frames:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
