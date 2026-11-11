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
    from v3_chan.experiment_metadata import (
        EXPERIMENT_METADATA_SCHEMA_VERSION,
        validate_experiment_metadata,
    )
    from v3_chan.hri_obs_recorder import HRIObsRecorder
    from v3_chan.pick_place_speed import (
        MOTION_EVENT_INDICES,
        SPEED_LIMITS_MPS,
        SPEED_SCHEDULE_VERSION,
        SPEED_STEP_SCHEMA_VERSION,
    )
except ImportError:
    from collection_provenance import validate_production_metadata
    from experiment_metadata import (
        EXPERIMENT_METADATA_SCHEMA_VERSION,
        validate_experiment_metadata,
    )
    from hri_obs_recorder import HRIObsRecorder
    from pick_place_speed import (
        MOTION_EVENT_INDICES,
        SPEED_LIMITS_MPS,
        SPEED_SCHEDULE_VERSION,
        SPEED_STEP_SCHEMA_VERSION,
    )


DEFAULT_MIN_HAND_POSE_VALID_FRACTION = 0.90
DEFAULT_MIN_RTF_VALID_FRACTION = 0.95
COMMAND_SPEED_ABS_TOLERANCE_MPS = 0.005
MEASURED_SPEED_ABS_TOLERANCE_MPS = 0.02
MEASURED_SPEED_REL_TOLERANCE = 0.05
APPLIED_XR_ANCHOR_STATUSES = {"xr_anchor", "xr_camera_teleport"}
GRASP_DATASET_PATHS = (
    "obs/has_grasped_cube",
    "hri_obs/has_grasped_cube",
    "task/has_grasped_cube",
)


@dataclass(frozen=True)
class ValidationReport:
    path: str
    episode_count: int
    layout_id: str
    speed_profiles: tuple[str, ...]
    hand_pose_valid_fraction: float
    left_hand_pose_valid_fraction: float
    right_hand_pose_valid_fraction: float
    real_time_factor_valid_fraction: float
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


def _check_grasp_signal(
    episode: h5py.Group,
    *,
    episode_name: str,
    length: int,
    require_success: bool,
    errors: list[str],
) -> None:
    signals: list[tuple[str, np.ndarray]] = []
    for dataset_name in GRASP_DATASET_PATHS:
        if dataset_name not in episode:
            continue
        values = np.asarray(episode[dataset_name][()]).reshape(-1)
        if values.size != length:
            errors.append(
                f"{episode_name}: {dataset_name} length={values.size}, expected {length}"
            )
            continue
        if not np.all(np.isfinite(values)):
            errors.append(f"{episode_name}: {dataset_name} contains non-finite values")
            continue
        if not np.all(np.isin(values, (0, 1, False, True))):
            errors.append(f"{episode_name}: {dataset_name} must contain only 0 or 1")
            continue
        signals.append((dataset_name, values > 0.5))

    if not signals:
        return
    if require_success and bool(episode.attrs.get("success", False)):
        if not any(np.any(values) for _, values in signals):
            errors.append(
                f"{episode_name}: successful episode has an all-zero grasp signal"
            )
    reference_name, reference = signals[0]
    for dataset_name, values in signals[1:]:
        if not np.array_equal(reference, values):
            errors.append(
                f"{episode_name}: grasp signals disagree between "
                f"{reference_name} and {dataset_name}"
            )


def _check_row_aligned_group(
    group: h5py.Group,
    *,
    group_name: str,
    episode_name: str,
    length: int,
    errors: list[str],
) -> None:
    def _visitor(name: str, item) -> None:
        if isinstance(item, h5py.Dataset):
            if item.ndim == 0 or item.shape[0] != length:
                errors.append(
                    f"{episode_name}: {group_name}/{name} length="
                    f"{item.shape[0] if item.ndim else 'scalar'}, expected {length}"
                )

    group.visititems(_visitor)


def _check_cartesian_speed_rows(
    episode: h5py.Group,
    *,
    episode_name: str,
    errors: list[str],
) -> None:
    required = (
        "controller/controller_event",
        "controller/nominal_phase_mode",
        "controller/speed_limit_mode",
        "controller/distance_gate",
        "controller/phase_speed_limit_mps",
        "controller/near_human_speed_limit_mps",
        "controller/blended_speed_limit_mps",
        "controller/active_speed_limit_mps",
        "controller/control_dt_valid",
        "controller/ee_speed_sim_mps",
        "controller/ee_speed_sim_valid",
        "controller/command_cartesian_speed_mps",
        "controller/command_limit_valid",
    )
    if any(name not in episode for name in required):
        return
    event = np.asarray(episode["controller/controller_event"][()], dtype=int)
    gate = np.asarray(episode["controller/distance_gate"][()], dtype=float)
    phase_limit = np.asarray(
        episode["controller/phase_speed_limit_mps"][()], dtype=float
    )
    near_limit = np.asarray(
        episode["controller/near_human_speed_limit_mps"][()], dtype=float
    )
    blended = np.asarray(
        episode["controller/blended_speed_limit_mps"][()], dtype=float
    )
    active = np.asarray(
        episode["controller/active_speed_limit_mps"][()], dtype=float
    )
    expected_blended = (1.0 - gate) * phase_limit + gate * near_limit
    motion = np.isin(event, MOTION_EVENT_INDICES)
    if np.any(motion & ~np.isclose(blended, expected_blended, atol=1e-6, rtol=0.0)):
        errors.append(
            f"{episode_name}: blended speed limit does not match the distance-gate formula"
        )
    if np.any(active > blended + 1e-6):
        errors.append(f"{episode_name}: active speed limit exceeds blended limit")

    command_valid = (
        np.asarray(episode["controller/control_dt_valid"][()], dtype=bool)
        & np.asarray(episode["controller/command_limit_valid"][()], dtype=bool)
        & motion
    )
    command_speed = np.asarray(
        episode["controller/command_cartesian_speed_mps"][()], dtype=float
    )
    if np.any(
        command_valid
        & (command_speed > active + COMMAND_SPEED_ABS_TOLERANCE_MPS)
    ):
        worst = float(np.max(command_speed[command_valid] - active[command_valid]))
        errors.append(
            f"{episode_name}: FK-predicted command exceeds active speed limit; "
            f"worst_excess_mps={worst:.6f}"
        )

    measured_valid = (
        np.asarray(episode["controller/ee_speed_sim_valid"][()], dtype=bool)
        & np.asarray(episode["controller/control_dt_valid"][()], dtype=bool)
        & motion
    )
    measured_speed = np.asarray(
        episode["controller/ee_speed_sim_mps"][()], dtype=float
    )
    tolerance = np.maximum(
        MEASURED_SPEED_ABS_TOLERANCE_MPS,
        MEASURED_SPEED_REL_TOLERANCE * active,
    )
    violation = measured_valid & (measured_speed > active + tolerance)
    if np.any(violation):
        worst = float(np.max(measured_speed[violation] - active[violation]))
        errors.append(
            f"{episode_name}: measured simulation-time EE speed exceeds active "
            f"limit tolerance; worst_excess_mps={worst:.6f}"
        )


def validate_hri_collection(
    path: str,
    *,
    expected_episodes: int | None = 3,
    require_success: bool = True,
    production: bool = True,
    min_hand_pose_valid_fraction: float = DEFAULT_MIN_HAND_POSE_VALID_FRACTION,
    min_rtf_valid_fraction: float = DEFAULT_MIN_RTF_VALID_FRACTION,
) -> ValidationReport:
    for name, value in (
        ("min_hand_pose_valid_fraction", min_hand_pose_valid_fraction),
        ("min_rtf_valid_fraction", min_rtf_valid_fraction),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    path = os.path.abspath(path)
    errors: list[str] = []
    hand_pose_valid_values: dict[str, list[np.ndarray]] = {
        "left_hand": [],
        "right_hand": [],
    }
    rtf_valid_values: list[np.ndarray] = []
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
                is_practice=_int(data.attrs.get("is_practice", -1)),
                experiment_condition=_text(
                    data.attrs.get("experiment_condition", "")
                ),
                experiment_block_id=_text(
                    data.attrs.get("experiment_block_id", "")
                ),
                haptic_condition=_text(
                    data.attrs.get("haptic_experiment_condition", "")
                ),
                haptics_enabled=_int(data.attrs.get("haptics_enabled", -1)),
                haptics_udp_configured=_int(
                    data.attrs.get("haptics_udp_configured", -1)
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
            "experiment_block_id",
            "collection_label",
            "haptics_transport",
            "haptics_enabled",
            "haptics_udp_configured",
            "haptics_intensity",
            "haptics_min_interval_s",
            "haptics_contact_min_steps",
            "haptics_pulse_flag_semantics",
            "xr_hand_sphere_enabled",
            "xr_anchor_status",
            "session_seed",
            "controller_speed_schedule_version",
            "controller_speed_step_schema_version",
            "controller_speed_profiles_json",
            "controller_speed_gate_start_gap_m",
            "controller_speed_gate_full_gap_m",
            "controller_speed_gate_blending",
            "controller_speed_acceleration_limit_mps2",
            "controller_speed_jerk_limit_mps3",
            "controller_speed_action_limit_method",
            "controller_speed_end_effector_reference",
            "controller_speed_end_effector_frame_name",
            "controller_speed_measured_pose_method",
        )
        for attr_name in required_file_attrs:
            if attr_name not in data.attrs:
                errors.append(f"missing root metadata attribute {attr_name}")
        metadata_schema = _text(
            data.attrs.get("experiment_metadata_schema_version", "")
        )
        if metadata_schema != EXPERIMENT_METADATA_SCHEMA_VERSION:
            errors.append(
                "experiment_metadata_schema_version="
                f"{metadata_schema!r}, expected {EXPERIMENT_METADATA_SCHEMA_VERSION!r}"
            )
        speed_schedule_version = _text(
            data.attrs.get("controller_speed_schedule_version", "")
        )
        if speed_schedule_version != SPEED_SCHEDULE_VERSION:
            errors.append(
                f"controller_speed_schedule_version={speed_schedule_version!r}, "
                f"expected {SPEED_SCHEDULE_VERSION!r}"
            )
        speed_step_schema = _text(
            data.attrs.get("controller_speed_step_schema_version", "")
        )
        if speed_step_schema != SPEED_STEP_SCHEMA_VERSION:
            errors.append(
                f"controller_speed_step_schema_version={speed_step_schema!r}, "
                f"expected {SPEED_STEP_SCHEMA_VERSION!r}"
            )
        if production and _text(
            data.attrs.get("controller_speed_end_effector_frame_name", "")
        ) in ("", "unknown"):
            errors.append(
                "controller_speed_end_effector_frame_name must name the actual Lula frame"
            )
        haptics_enabled_value = _int(data.attrs.get("haptics_enabled", -1))
        if "haptics_intensity" in data.attrs:
            intensity = _int(data.attrs["haptics_intensity"], -1)
            if not 0 <= intensity <= 100:
                errors.append("haptics_intensity must be in [0, 100]")
        if "haptics_contact_min_steps" in data.attrs and _int(
            data.attrs["haptics_contact_min_steps"], 0
        ) < 1:
            errors.append("haptics_contact_min_steps must be >= 1")
        if "haptics_min_interval_s" in data.attrs:
            try:
                haptics_min_interval_s = float(data.attrs["haptics_min_interval_s"])
            except (TypeError, ValueError):
                haptics_min_interval_s = float("nan")
            if not np.isfinite(haptics_min_interval_s) or haptics_min_interval_s < 0.0:
                errors.append("haptics_min_interval_s must be finite and >= 0")
        xr_anchor_status = _text(data.attrs.get("xr_anchor_status", ""))
        if production and xr_anchor_status not in APPLIED_XR_ANCHOR_STATUSES:
            errors.append(
                "xr_anchor_status must indicate an applied XR anchor; "
                f"found={xr_anchor_status!r}"
            )

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
            "safety/haptic_pulse_left",
            "safety/haptic_pulse_right",
            "controller/controller_event",
            "controller/controller_speed_profile",
            "controller/nominal_phase_mode",
            "controller/speed_limit_mode",
            "controller/distance_gate",
            "controller/active_speed_limit_mps",
            "controller/ee_velocity_sim_world_mps",
            "controller/ee_speed_sim_mps",
            "controller/ee_speed_sim_valid",
            "controller/ee_velocity_wall_world_mps",
            "controller/ee_speed_wall_mps",
            "controller/ee_speed_wall_valid",
            "controller/command_cartesian_speed_mps",
            "controller/command_limit_valid",
        )
        for episode_name in episode_names:
            episode = episodes_group[episode_name]
            length = int(episode.attrs.get("episode_length", -1))
            if length <= 0:
                errors.append(f"{episode_name}: invalid episode_length={length}")
                continue
            if require_success and not bool(episode.attrs.get("success", False)):
                errors.append(f"{episode_name}: success is false")
            _check_grasp_signal(
                episode,
                episode_name=episode_name,
                length=length,
                require_success=require_success,
                errors=errors,
            )
            speed_profile = _text(
                episode.attrs.get("controller_speed_profile", "")
            )
            speed_profiles.append(speed_profile)
            expected_limits = SPEED_LIMITS_MPS.get(speed_profile)
            if expected_limits is None:
                errors.append(
                    f"{episode_name}: unknown controller_speed_profile={speed_profile!r}"
                )
            else:
                for attr_name, limit_name in (
                    ("transport_speed_limit_mps", "transport"),
                    ("approach_place_speed_limit_mps", "approach_place"),
                    ("near_human_speed_limit_mps", "near_human"),
                ):
                    try:
                        actual_limit = float(episode.attrs[attr_name])
                    except (KeyError, TypeError, ValueError):
                        errors.append(
                            f"{episode_name}: missing or invalid {attr_name}"
                        )
                        continue
                    if not np.isclose(
                        actual_limit,
                        expected_limits[limit_name],
                        atol=1e-9,
                        rtol=0.0,
                    ):
                        errors.append(
                            f"{episode_name}: {attr_name}={actual_limit}, expected "
                            f"{expected_limits[limit_name]} for {speed_profile}"
                        )
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
            for row_group_name in (
                "obs",
                "hri_obs",
                "human",
                "safety",
                "dynamic_sim",
                "controller",
                "errp",
                "actions",
                "rewards",
                "task",
            ):
                if row_group_name in episode:
                    _check_row_aligned_group(
                        episode[row_group_name],
                        group_name=row_group_name,
                        episode_name=episode_name,
                        length=length,
                        errors=errors,
                    )
            _check_cartesian_speed_rows(
                episode,
                episode_name=episode_name,
                errors=errors,
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
                "controller/distance_gate",
                "controller/active_speed_limit_mps",
                "controller/ee_velocity_sim_world_mps",
                "controller/ee_speed_sim_mps",
                "controller/ee_velocity_wall_world_mps",
                "controller/ee_speed_wall_mps",
                "controller/command_cartesian_speed_mps",
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
                raw_rtf_valid = np.asarray(
                    episode["real_time_factor_valid"][()]
                ).reshape(-1)
                if not np.all(np.isin(raw_rtf_valid, (0, 1, False, True))):
                    errors.append(
                        f"{episode_name}: real_time_factor_valid must contain only 0 or 1"
                    )
                rtf_valid_fraction = float(rtf_valid.mean()) if rtf_valid.size else 0.0
                rtf_valid_values.append(rtf_valid.reshape(-1))
                if production and rtf_valid_fraction < float(min_rtf_valid_fraction):
                    errors.append(
                        f"{episode_name}: real_time_factor_valid fraction="
                        f"{rtf_valid_fraction:.4f}, required >= "
                        f"{float(min_rtf_valid_fraction):.4f}"
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
            for pulse_name in (
                "safety/haptic_pulse_left",
                "safety/haptic_pulse_right",
            ):
                if pulse_name not in episode:
                    continue
                pulses = np.asarray(episode[pulse_name][()]).reshape(-1)
                if not np.all(np.isin(pulses, (0, 1, False, True))):
                    errors.append(
                        f"{episode_name}: {pulse_name} must contain only 0 or 1"
                    )
                if haptics_enabled_value == 0 and np.any(pulses != 0):
                    errors.append(
                        f"{episode_name}: {pulse_name} is nonzero while haptics are disabled"
                    )
            for hand_name in ("left_hand", "right_hand"):
                pose_name = f"human/{hand_name}_pose_valid"
                if pose_name in episode:
                    pose_valid = np.asarray(
                        episode[pose_name][()], dtype=float
                    ).reshape(-1)
                    if not np.all(np.isfinite(pose_valid)) or not np.all(
                        np.isin(pose_valid, (0.0, 1.0))
                    ):
                        errors.append(
                            f"{episode_name}: {pose_name} must contain finite 0/1 values"
                        )
                        continue
                    pose_valid_fraction = (
                        float(pose_valid.mean()) if pose_valid.size else 0.0
                    )
                    hand_pose_valid_values[hand_name].append(pose_valid)
                    if production and pose_valid_fraction < float(
                        min_hand_pose_valid_fraction
                    ):
                        errors.append(
                            f"{episode_name}: {pose_name} valid fraction="
                            f"{pose_valid_fraction:.4f}, required >= "
                            f"{float(min_hand_pose_valid_fraction):.4f}"
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
    valid_pose_values = [
        values
        for hand_values in hand_pose_valid_values.values()
        for values in hand_values
        if values.size
    ]
    hand_pose_valid_fraction = (
        float(np.concatenate(valid_pose_values).mean()) if valid_pose_values else 0.0
    )
    per_hand_pose_valid_fraction = {
        hand_name: (
            float(np.concatenate(values).mean())
            if any(value.size for value in values)
            else 0.0
        )
        for hand_name, values in hand_pose_valid_values.items()
    }
    nonempty_rtf_valid_values = [values for values in rtf_valid_values if values.size]
    real_time_factor_valid_fraction = (
        float(np.concatenate(nonempty_rtf_valid_values).mean())
        if nonempty_rtf_valid_values
        else 0.0
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
        left_hand_pose_valid_fraction=per_hand_pose_valid_fraction["left_hand"],
        right_hand_pose_valid_fraction=per_hand_pose_valid_fraction["right_hand"],
        real_time_factor_valid_fraction=real_time_factor_valid_fraction,
        median_real_time_factor=median_real_time_factor,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--expected-episodes", type=int, default=3)
    parser.add_argument("--allow-failed-episode", action="store_true")
    parser.add_argument("--non-production", action="store_true")
    parser.add_argument(
        "--min-hand-pose-valid-fraction",
        type=float,
        default=DEFAULT_MIN_HAND_POSE_VALID_FRACTION,
    )
    parser.add_argument(
        "--min-rtf-valid-fraction",
        type=float,
        default=DEFAULT_MIN_RTF_VALID_FRACTION,
    )
    args = parser.parse_args()
    report = validate_hri_collection(
        args.path,
        expected_episodes=args.expected_episodes,
        require_success=not args.allow_failed_episode,
        production=not args.non_production,
        min_hand_pose_valid_fraction=args.min_hand_pose_valid_fraction,
        min_rtf_valid_fraction=args.min_rtf_valid_fraction,
    )
    print(
        "[ValidateHRI] valid "
        f"path={report.path} episodes={report.episode_count} "
        f"layout_id={report.layout_id} speeds={','.join(report.speed_profiles)} "
        f"hand_pose_valid_fraction={report.hand_pose_valid_fraction:.4f} "
        f"left={report.left_hand_pose_valid_fraction:.4f} "
        f"right={report.right_hand_pose_valid_fraction:.4f} "
        f"rtf_valid_fraction={report.real_time_factor_valid_fraction:.4f} "
        f"median_rtf={report.median_real_time_factor:.4f}"
    )


if __name__ == "__main__":
    main()
