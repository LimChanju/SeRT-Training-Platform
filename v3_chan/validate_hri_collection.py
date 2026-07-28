#!/usr/bin/env python3
"""Validate one production HRI trajectory before it enters training."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import h5py
import numpy as np

try:
    from v3_chan.collection_provenance import validate_production_metadata
    from v3_chan.experiment_metadata import validate_experiment_metadata
    from v3_chan.hri_obs_recorder import HRIObsRecorder
except ImportError:
    from collection_provenance import validate_production_metadata
    from experiment_metadata import validate_experiment_metadata
    from hri_obs_recorder import HRIObsRecorder


@dataclass(frozen=True)
class ValidationReport:
    path: str
    episode_count: int
    layout_id: str
    speed_profiles: tuple[str, ...]
    hand_pose_valid_fraction: float
    median_real_time_factor: float


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _text(value.item())
    return str(value)


def _int(value, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _check_finite_and_monotonic(
    values,
    *,
    dataset_name: str,
    episode_name: str,
    errors: list[str],
    require_monotonic: bool = False,
) -> None:
    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        errors.append(f"{episode_name}: {dataset_name} contains non-finite values")
        return
    scalar_series = array.reshape(-1)
    if (
        require_monotonic
        and scalar_series.size > 1
        and np.any(np.diff(scalar_series) < 0)
    ):
        errors.append(f"{episode_name}: {dataset_name} is not monotonic")


def validate_hri_collection(
    path: str,
    *,
    expected_episodes: int | None = 3,
    require_success: bool = True,
    production: bool = True,
) -> ValidationReport:
    path = os.path.abspath(path)
    errors: list[str] = []
    hand_pose_valid_values: list[np.ndarray] = []
    valid_real_time_factors: list[np.ndarray] = []
    with h5py.File(path, "r") as data:
        schema = _text(data.attrs.get("schema_version", ""))
        if schema != HRIObsRecorder.SCHEMA_VERSION:
            errors.append(
                f"schema_version={schema!r}, expected {HRIObsRecorder.SCHEMA_VERSION!r}"
            )
        try:
            validate_production_metadata(
                production_mode=production,
                participant_id=_text(data.attrs.get("participant_id", "")),
                code_version=_text(data.attrs.get("code_version", "")),
            )
        except RuntimeError as exc:
            errors.append(str(exc))
        try:
            validate_experiment_metadata(
                production_mode=production,
                participant_id=_text(data.attrs.get("participant_id", "")),
                participant_session_index=_int(
                    data.attrs.get("participant_session_index", -1)
                ),
                participant_handedness=_text(
                    data.attrs.get("participant_handedness", "")
                ),
                experiment_condition=_text(
                    data.attrs.get("experiment_condition", "")
                ),
                haptic_condition=_text(
                    data.attrs.get("haptic_experiment_condition", "")
                ),
                protocol_version=_text(
                    data.attrs.get("collection_protocol_version", "")
                ),
                room_calibration_id=_text(
                    data.attrs.get("room_calibration_id", "")
                ),
            )
        except RuntimeError as exc:
            errors.append(str(exc))
        source_hash = _text(data.attrs.get("source_tree_sha256", ""))
        if production and (
            len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash.lower())
        ):
            errors.append("source_tree_sha256 must be a 64-character hexadecimal hash")
        required_file_attrs = (
            "experiment_metadata_schema_version",
            "participant_id_kind",
            "participant_session_index",
            "participant_handedness",
            "is_practice",
            "collection_label",
            "haptics_transport",
            "haptics_intensity",
            "haptics_min_interval_s",
            "haptics_contact_min_steps",
            "haptics_pulse_flag_semantics",
            "xr_hand_sphere_enabled",
            "xr_anchor_status",
            "session_seed",
        )
        for attr_name in required_file_attrs:
            if attr_name not in data.attrs:
                errors.append(f"missing root metadata attribute {attr_name}")
        if "haptics_intensity" in data.attrs:
            intensity = _int(data.attrs["haptics_intensity"], -1)
            if not 0 <= intensity <= 100:
                errors.append("haptics_intensity must be in [0, 100]")
        if "haptics_contact_min_steps" in data.attrs and _int(
            data.attrs["haptics_contact_min_steps"], 0
        ) < 1:
            errors.append("haptics_contact_min_steps must be >= 1")

        episodes_group = data.get("episodes")
        episode_names = tuple(sorted(episodes_group.keys())) if episodes_group else ()
        if expected_episodes is not None and len(episode_names) != int(expected_episodes):
            errors.append(
                f"episode_count={len(episode_names)}, expected {int(expected_episodes)}"
            )

        layout_ids = []
        initial_cube_positions = []
        initial_cube_orientations = []
        initial_target_positions = []
        speed_profiles = []
        required_row_datasets = (
            "sim_time",
            "monotonic_time_ns",
            "pose_monotonic_time_ns",
            "wall_time_unix_ns",
            "real_time_factor",
            "real_time_factor_valid",
            "action_command_monotonic_ns",
            "obs_policy",
            "hri_obs_policy",
            "human_valid_mask",
            "human/left_hand_pose_source_id",
            "human/right_hand_pose_source_id",
            "human/left_hand_position_tracked",
            "human/right_hand_position_tracked",
            "human/left_hand_pose_valid",
            "human/right_hand_pose_valid",
            "human/left_hand_pos",
            "human/right_hand_pos",
            "actions/previous_applied_joint_positions",
            "actions/next_commanded_joint_positions",
            "dynamic_sim/left_ttc_s",
            "dynamic_sim/right_ttc_s",
            "safety/left_ttc_s",
            "safety/right_ttc_s",
            "human/left_hand_vel_filtered_mps",
            "human/right_hand_vel_filtered_mps",
        )
        for episode_name in episode_names:
            episode = episodes_group[episode_name]
            length = int(episode.attrs.get("episode_length", -1))
            if length <= 0:
                errors.append(f"{episode_name}: invalid episode_length={length}")
                continue
            if require_success and not bool(episode.attrs.get("success", False)):
                errors.append(f"{episode_name}: success is false")
            speed_profiles.append(_text(episode.attrs.get("controller_speed_profile", "")))
            if "initial_scene/layout_id" not in episode:
                errors.append(f"{episode_name}: missing initial_scene/layout_id")
            else:
                layout_ids.append(_text(episode["initial_scene/layout_id"][()]))
            if "initial_scene/cube_positions_world" not in episode:
                errors.append(f"{episode_name}: missing initial cube positions")
            elif episode["initial_scene/cube_positions_world"].shape != (6, 3):
                errors.append(
                    f"{episode_name}: cube_positions_world shape="
                    f"{episode['initial_scene/cube_positions_world'].shape}, expected (6, 3)"
                )
            else:
                initial_cube_positions.append(
                    np.asarray(
                        episode["initial_scene/cube_positions_world"][()], dtype=float
                    )
                )
            if "initial_scene/cube_orientations_wxyz" not in episode:
                errors.append(f"{episode_name}: missing initial cube orientations")
            elif episode["initial_scene/cube_orientations_wxyz"].shape != (6, 4):
                errors.append(
                    f"{episode_name}: cube_orientations_wxyz shape="
                    f"{episode['initial_scene/cube_orientations_wxyz'].shape}, expected (6, 4)"
                )
            else:
                initial_cube_orientations.append(
                    np.asarray(
                        episode["initial_scene/cube_orientations_wxyz"][()],
                        dtype=float,
                    )
                )
            if "initial_scene/place_target_position_world" not in episode:
                errors.append(f"{episode_name}: missing initial place target position")
            else:
                initial_target_positions.append(
                    np.asarray(
                        episode["initial_scene/place_target_position_world"][()],
                        dtype=float,
                    )
                )
            for dataset_name in required_row_datasets:
                if dataset_name not in episode:
                    errors.append(f"{episode_name}: missing {dataset_name}")
                    continue
                if episode[dataset_name].shape[0] != length:
                    errors.append(
                        f"{episode_name}: {dataset_name} length="
                        f"{episode[dataset_name].shape[0]}, expected {length}"
                    )
            for dataset_name in (
                "sim_time",
                "monotonic_time_ns",
                "pose_monotonic_time_ns",
                "wall_time_unix_ns",
                "action_command_monotonic_ns",
            ):
                if dataset_name in episode:
                    _check_finite_and_monotonic(
                        episode[dataset_name][()],
                        dataset_name=dataset_name,
                        episode_name=episode_name,
                        errors=errors,
                        require_monotonic=True,
                    )
            for dataset_name in (
                "real_time_factor",
                "obs_policy",
                "hri_obs_policy",
                "human_valid_mask",
                "human/left_hand_pos",
                "human/right_hand_pos",
                "safety/left_ttc_s",
                "safety/right_ttc_s",
                "dynamic_sim/left_ttc_s",
                "dynamic_sim/right_ttc_s",
            ):
                if dataset_name in episode:
                    _check_finite_and_monotonic(
                        episode[dataset_name][()],
                        dataset_name=dataset_name,
                        episode_name=episode_name,
                        errors=errors,
                    )
            if "real_time_factor" in episode and "real_time_factor_valid" in episode:
                rtf = np.asarray(episode["real_time_factor"][()], dtype=float)
                rtf_valid = np.asarray(
                    episode["real_time_factor_valid"][()], dtype=bool
                )
                if np.any(rtf_valid & (rtf <= 0.0)):
                    errors.append(
                        f"{episode_name}: valid real_time_factor must be positive"
                    )
                valid_real_time_factors.append(rtf[rtf_valid])
            for tracked_name in (
                "human/head_position_tracked",
                "human/left_hand_position_tracked",
                "human/right_hand_position_tracked",
            ):
                if tracked_name in episode:
                    values = set(
                        np.asarray(episode[tracked_name][()], dtype=int).tolist()
                    )
                    if not values <= {-1, 0, 1}:
                        errors.append(
                            f"{episode_name}: {tracked_name} contains invalid values {values}"
                        )
            for hand_name in ("left_hand", "right_hand"):
                pose_name = f"human/{hand_name}_pose_valid"
                if pose_name in episode:
                    hand_pose_valid_values.append(
                        np.asarray(episode[pose_name][()], dtype=float).reshape(-1)
                    )

        unique_layouts = tuple(sorted(set(layout_ids)))
        if episode_names and len(unique_layouts) != 1:
            errors.append(
                "all speed conditions in one session must share exactly one layout_id; "
                f"found={unique_layouts}"
            )
        for label, values in (
            ("cube positions", initial_cube_positions),
            ("cube orientations", initial_cube_orientations),
            ("place target positions", initial_target_positions),
        ):
            if values and any(
                not np.allclose(values[0], value, atol=1e-6, rtol=0.0)
                for value in values[1:]
            ):
                errors.append(
                    f"initial {label} differ across speed conditions despite shared layout_id"
                )
        if expected_episodes == 3 and set(speed_profiles) != {"slow", "medium", "fast"}:
            errors.append(
                "three-episode production session must contain slow, medium, and fast; "
                f"found={speed_profiles}"
            )

    if errors:
        raise ValueError("HRI collection validation failed:\n- " + "\n- ".join(errors))
    valid_pose_values = [values for values in hand_pose_valid_values if values.size]
    hand_pose_valid_fraction = (
        float(np.concatenate(valid_pose_values).mean()) if valid_pose_values else 0.0
    )
    valid_rtf_values = [values for values in valid_real_time_factors if values.size]
    median_real_time_factor = (
        float(np.median(np.concatenate(valid_rtf_values)))
        if valid_rtf_values
        else 0.0
    )
    return ValidationReport(
        path=path,
        episode_count=len(episode_names),
        layout_id=layout_ids[0] if layout_ids else "",
        speed_profiles=tuple(speed_profiles),
        hand_pose_valid_fraction=hand_pose_valid_fraction,
        median_real_time_factor=median_real_time_factor,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--expected-episodes", type=int, default=3)
    parser.add_argument("--allow-failed-episode", action="store_true")
    parser.add_argument("--non-production", action="store_true")
    args = parser.parse_args()
    report = validate_hri_collection(
        args.path,
        expected_episodes=args.expected_episodes,
        require_success=not args.allow_failed_episode,
        production=not args.non_production,
    )
    print(
        "[ValidateHRI] valid "
        f"path={report.path} episodes={report.episode_count} "
        f"layout_id={report.layout_id} speeds={','.join(report.speed_profiles)} "
        f"hand_pose_valid_fraction={report.hand_pose_valid_fraction:.4f} "
        f"median_rtf={report.median_real_time_factor:.4f}"
    )


if __name__ == "__main__":
    main()
