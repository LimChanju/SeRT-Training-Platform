from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import h5py
import numpy as np


MANIFEST_VERSION = "hri_encounter_manifest_v2"
SOURCE_CONFIGURATION_VERSION = "hri_source_configuration_v1"
SEVERITY_ORDER = ("safe", "gate_only", "near", "near_miss", "collision")
TASK_PHASE_NAMES = (
    "approach_cube",
    "grasp_cube",
    "move_to_target",
    "release_cube",
)


@dataclass(frozen=True)
class EncounterBuildConfig:
    gate_start_m: float = 0.13
    near_m: float = 0.05
    near_miss_m: float = 0.02
    collision_m: float = 0.0
    clear_end_m: float = 0.15
    onset_frames: int = 3
    clear_frames: int = 15
    margin_frames: int = 30
    safe_window_frames: int = 180
    safe_stride_frames: int = 90
    safe_min_frames: int = 120
    max_safe_per_episode: int = 8


def build_encounter_manifest(
    source_paths: Sequence[str],
    output_path: str,
    *,
    config: EncounterBuildConfig | None = None,
) -> dict[str, Any]:
    cfg = config or EncounterBuildConfig()
    _validate_config(cfg)
    sources = tuple(os.path.abspath(os.path.expanduser(path)) for path in source_paths)
    if not sources:
        raise ValueError("At least one source HDF5 path is required.")

    scenarios: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source_path in sources:
        if not os.path.exists(source_path):
            raise FileNotFoundError(source_path)
        with h5py.File(source_path, "r") as h5_file:
            if "episodes" not in h5_file:
                raise KeyError(f"No /episodes group in {source_path}")
            episode_names = tuple(sorted(h5_file["episodes"].keys()))
            source_start = len(scenarios)
            for episode_name in episode_names:
                scenarios.extend(
                    _build_episode_scenarios(
                        source_path,
                        episode_name,
                        h5_file["episodes"][episode_name],
                        cfg,
                    )
                )
            source_summaries.append(
                {
                    "path": source_path,
                    "episode_count": len(episode_names),
                    "scenario_count": len(scenarios) - source_start,
                }
            )

    counts = {
        severity: sum(
            scenario["target_severity"] == severity for scenario in scenarios
        )
        for severity in SEVERITY_ORDER
    }
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "source_configuration_schema_version": SOURCE_CONFIGURATION_VERSION,
        "phase_anchor_policy": "minimum_surface_gap_in_core",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_config": asdict(cfg),
        "sources": source_summaries,
        "scenario_count": len(scenarios),
        "severity_counts": counts,
        "scenarios": scenarios,
    }
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return manifest


def load_encounter_manifest(path: str) -> dict[str, Any]:
    manifest_path = os.path.abspath(os.path.expanduser(path))
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported encounter manifest version: "
            f"{manifest.get('schema_version')!r}"
        )
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"Encounter manifest contains no scenarios: {manifest_path}")
    manifest["_manifest_path"] = manifest_path
    return manifest


def parse_severity_mix(value: str | dict[str, float] | None) -> dict[str, float]:
    if value is None or value == "":
        return {
            "safe": 0.40,
            "gate_only": 0.25,
            "near": 0.20,
            "near_miss": 0.10,
            "collision": 0.05,
        }
    if isinstance(value, dict):
        parsed = {str(key): float(weight) for key, weight in value.items()}
    else:
        parsed: dict[str, float] = {}
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(
                    "Encounter severity mix must use name=weight entries."
                )
            name, raw_weight = item.split("=", 1)
            parsed[name.strip()] = float(raw_weight)
    unknown = sorted(set(parsed) - set(SEVERITY_ORDER))
    if unknown:
        raise ValueError(f"Unknown encounter severities: {unknown}")
    if any(weight < 0.0 for weight in parsed.values()):
        raise ValueError("Encounter severity weights must be non-negative.")
    total = float(sum(parsed.values()))
    if total <= 0.0:
        raise ValueError("Encounter severity mix must have a positive total weight.")
    return {
        severity: float(parsed.get(severity, 0.0)) / total
        for severity in SEVERITY_ORDER
    }


def resolve_scenario_source(
    scenario: dict[str, Any],
    manifest_path: str,
) -> str:
    source_path = os.path.expanduser(str(scenario["source_path"]))
    candidates = [source_path]
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    if not os.path.isabs(source_path):
        candidates.append(os.path.join(manifest_dir, source_path))
    candidates.extend(
        (
            os.path.join(manifest_dir, os.path.basename(source_path)),
            os.path.join(
                os.path.dirname(manifest_dir), os.path.basename(source_path)
            ),
        )
    )
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Source HDF5 for encounter {scenario.get('id', '')} was not found: "
        f"{source_path}"
    )


def extract_episode_source_configuration(
    group: h5py.Group,
    *,
    active_cube_index: int,
    source_anchor_step: int,
) -> dict[str, Any]:
    """Extract reproducible scene provenance from one recorded episode.

    Current v8 collections contain exact cube and target poses. Older files are
    supported by recording which fields are absent and, where possible, using
    the first observation for the robot's initial joint state.
    """

    provenance: dict[str, str] = {}

    def value(paths: Sequence[str], *, attr_names: Sequence[str] = ()) -> Any:
        for path in paths:
            if path in group:
                provenance[paths[0]] = path
                return group[path][()]
        for attr_name in attr_names:
            if attr_name in group.attrs:
                provenance[paths[0]] = f"@{attr_name}"
                return group.attrs[attr_name]
        return None

    cube_names = _string_list(
        value(("initial_scene/cube_names",)),
    )
    cube_roles = _string_list(
        value(("initial_scene/cube_roles",)),
    )
    cube_positions = _finite_array_or_none(
        value(("initial_scene/cube_positions_world",)),
        width=3,
    )
    cube_orientations = _finite_array_or_none(
        value(("initial_scene/cube_orientations_wxyz",)),
        width=4,
    )
    target_position = _finite_vector_or_none(
        value(("initial_scene/place_target_position_world",)),
        width=3,
    )
    target_orientation = _finite_vector_or_none(
        value(("initial_scene/place_target_orientation_wxyz",)),
        width=4,
    )

    robot_initial_joint_positions = _finite_vector_or_none(
        value(
            (
                "initial_scene/robot_joint_positions",
                "initial_scene/robot_joint_pos",
            )
        ),
    )
    if robot_initial_joint_positions is None:
        robot_initial_joint_positions = _step_vector_or_none(
            group,
            ("obs/robot_joint_pos",),
            0,
        )
        if robot_initial_joint_positions is not None:
            provenance["initial_scene/robot_joint_positions"] = (
                "obs/robot_joint_pos[0]"
            )
    robot_initial_joint_positions = _complete_panda_joint_positions(
        group,
        robot_initial_joint_positions,
        step=0,
    )
    robot_initial_joint_velocities = _finite_vector_or_none(
        value(
            (
                "initial_scene/robot_joint_velocities",
                "initial_scene/robot_joint_vel",
            )
        ),
    )
    if robot_initial_joint_velocities is None:
        robot_initial_joint_velocities = _step_vector_or_none(
            group,
            ("obs/robot_joint_vel",),
            0,
        )
        if robot_initial_joint_velocities is not None:
            provenance["initial_scene/robot_joint_velocities"] = (
                "obs/robot_joint_vel[0]"
            )
    robot_initial_joint_velocities = _complete_panda_joint_velocities(
        robot_initial_joint_velocities,
        robot_initial_joint_positions,
    )

    source_anchor_ee_pos = _step_vector_or_none(
        group,
        ("obs/ee_pos",),
        source_anchor_step,
        width=3,
    )
    source_anchor_ee_quat = _step_vector_or_none(
        group,
        ("obs/ee_quat",),
        source_anchor_step,
        width=4,
    )
    source_anchor_robot_joint_positions = _step_vector_or_none(
        group,
        ("obs/robot_joint_pos",),
        source_anchor_step,
    )
    source_anchor_robot_joint_velocities = _step_vector_or_none(
        group,
        ("obs/robot_joint_vel",),
        source_anchor_step,
    )
    source_anchor_robot_joint_positions = _complete_panda_joint_positions(
        group,
        source_anchor_robot_joint_positions,
        step=source_anchor_step,
    )
    source_anchor_robot_joint_velocities = _complete_panda_joint_velocities(
        source_anchor_robot_joint_velocities,
        source_anchor_robot_joint_positions,
    )

    collection_seed = _int_or_none(
        value(
            ("initial_scene/session_seed",),
            attr_names=("session_seed", "collection_seed"),
        )
    )
    layout_seed = _int_or_none(
        value(
            ("initial_scene/layout_seed",),
            attr_names=("layout_seed",),
        )
    )
    layout_id = _text_or_none(
        value(
            ("initial_scene/layout_id",),
            attr_names=("layout_id",),
        )
    )
    measured_layout_id = _text_or_none(
        value(("initial_scene/measured_layout_id",))
    )

    required_exact = {
        "cube_names": cube_names,
        "cube_positions_world": cube_positions,
        "cube_orientations_wxyz": cube_orientations,
        "place_target_position_world": target_position,
        "place_target_orientation_wxyz": target_orientation,
    }
    missing_fields = [name for name, item in required_exact.items() if item is None]
    cube_count = 0 if cube_positions is None else int(cube_positions.shape[0])
    if cube_orientations is not None and cube_orientations.shape[0] != cube_count:
        missing_fields.append("cube_pose_row_count_match")
    if cube_names is not None and len(cube_names) != cube_count:
        missing_fields.append("cube_name_count_match")
    if not 0 <= int(active_cube_index) < cube_count:
        missing_fields.append("active_cube_index_in_range")

    source_cube_name = None
    if cube_names is not None and 0 <= int(active_cube_index) < len(cube_names):
        source_cube_name = cube_names[int(active_cube_index)]

    return {
        "schema_version": SOURCE_CONFIGURATION_VERSION,
        "episode_name": group.name.rsplit("/", 1)[-1],
        "active_cube_index": int(active_cube_index),
        "active_cube_name": source_cube_name,
        "collection_seed": collection_seed,
        "layout_seed": layout_seed,
        "layout_id": layout_id,
        "measured_layout_id": measured_layout_id,
        "cube_names": cube_names,
        "cube_roles": cube_roles,
        "cube_positions_world": _array_list(cube_positions),
        "cube_orientations_wxyz": _array_list(cube_orientations),
        "place_target_position_world": _array_list(target_position),
        "place_target_orientation_wxyz": _array_list(target_orientation),
        "robot_initial_joint_positions": _array_list(
            robot_initial_joint_positions
        ),
        "robot_initial_joint_velocities": _array_list(
            robot_initial_joint_velocities
        ),
        "source_anchor_step": int(source_anchor_step),
        "source_anchor_ee_pos": _array_list(source_anchor_ee_pos),
        "source_anchor_ee_quat_wxyz": _array_list(source_anchor_ee_quat),
        "source_anchor_robot_joint_positions": _array_list(
            source_anchor_robot_joint_positions
        ),
        "source_anchor_robot_joint_velocities": _array_list(
            source_anchor_robot_joint_velocities
        ),
        "exact_pose_available": not missing_fields,
        "collection_seed_available": layout_seed is not None,
        "missing_fields": sorted(set(missing_fields)),
        "provenance": provenance,
    }


def resolve_source_restoration(
    scenario: dict[str, Any],
    *,
    screening_seed: int,
    allow_legacy_fallback: bool = False,
) -> dict[str, Any]:
    """Choose the safest available source-scene restoration strategy."""

    source = scenario.get("source_configuration")
    source = dict(source) if isinstance(source, dict) else {}
    source_cube_index = _int_or_none(
        source.get("active_cube_index", scenario.get("cube_index"))
    )
    base = {
        "source_configuration_schema_version": source.get("schema_version", ""),
        "source_configuration_available": False,
        "restoration_mode": "unavailable",
        "restoration_reason": "source_configuration_missing",
        "source_cube_index": source_cube_index,
        "screening_cube_index": None,
        "source_cube_name": source.get("active_cube_name"),
        "collection_seed": _int_or_none(source.get("collection_seed")),
        "layout_seed": _int_or_none(source.get("layout_seed")),
        "screening_seed": int(screening_seed),
        "source_layout_id": source.get("layout_id"),
        "source_measured_layout_id": source.get("measured_layout_id"),
        "cube_pose_restored": False,
        "target_pose_restored": False,
        "robot_initial_state_restored": False,
        "pose_mismatch": False,
        "pose_mismatch_reason": "",
        "missing_fields": list(source.get("missing_fields", [])),
        "source_configuration": source,
    }
    if source_cube_index is None or source_cube_index < 0:
        base["restoration_reason"] = "active_cube_index_missing"
        return base

    if bool(source.get("exact_pose_available", False)):
        base.update(
            {
                "source_configuration_available": True,
                "restoration_mode": "exact_pose",
                "restoration_reason": "",
            }
        )
        return base

    layout_seed = _int_or_none(source.get("layout_seed"))
    if layout_seed is not None:
        base.update(
            {
                "source_configuration_available": True,
                "restoration_mode": "collection_seed",
                "restoration_reason": "exact_pose_unavailable",
                "layout_seed": layout_seed,
            }
        )
        return base

    if allow_legacy_fallback:
        base.update(
            {
                "source_configuration_available": True,
                "restoration_mode": "legacy_fallback",
                "restoration_reason": "explicit_legacy_fallback",
            }
        )
        return base

    if source:
        base["restoration_reason"] = "source_configuration_incomplete"
    return base


def _build_episode_scenarios(
    source_path: str,
    episode_name: str,
    group: h5py.Group,
    cfg: EncounterBuildConfig,
) -> list[dict[str, Any]]:
    length = _episode_length(group)
    if length <= 0:
        return []

    gap = _read_scalar(
        group,
        (
            "safety/min_hand_end_effector_surface_gap_m",
            "safety/end_effector_surface_gap_m",
            "obs/min_hand_end_effector_surface_gap",
            "safety/min_hand_gripper_surface_gap_m",
            "obs/min_hand_gripper_surface_gap",
            "safety/min_hand_gripper_dist_m",
            "obs/min_hand_gripper_dist",
        ),
        length,
        fill=np.inf,
    )
    contact = _read_scalar(
        group,
        (
            "safety/contact_active",
            "safety/human_robot_collision",
            "obs/human_robot_collision",
        ),
        length,
        fill=0.0,
    ) > 0.5
    cube_index = _read_int(
        group,
        ("task/current_pick_idx", "current_pick_idx", "obs/current_pick_idx"),
        length,
        fill=-1,
    )
    attempt_index = _read_int(
        group,
        (
            "task/attempt_index",
            "task/current_cube_attempt",
            "current_cube_attempt",
        ),
        length,
        fill=0,
    )
    controller_event = _read_int(
        group,
        ("task/controller_event", "obs/controller_event_index"),
        length,
        fill=-1,
    )
    task_phase = _read_task_phase(group, controller_event, length)
    ee_pos = _read_vector(group, ("obs/ee_pos",), length, width=3)
    valid_mask = _read_vector(
        group,
        ("human/valid_mask", "human_valid_mask"),
        length,
        width=3,
    )
    if valid_mask is None:
        left = _read_vector(
            group,
            ("human/left_hand_pos", "obs/human_left_hand_pos"),
            length,
            width=3,
        )
        right = _read_vector(
            group,
            ("human/right_hand_pos", "obs/human_right_hand_pos"),
            length,
            width=3,
        )
        hand_valid = _valid_hand_series(left, right, length)
    else:
        hand_valid = np.any(valid_mask[:, 1:3] > 0.5, axis=1)

    closest_link = _read_int(
        group,
        (
            "safety/closest_robot_link_id",
            "safety/closest_end_effector_link_id",
        ),
        length,
        fill=-1,
    )
    closest_hand = _read_int(
        group,
        ("safety/closest_human_hand_id", "safety/closest_hand_id"),
        length,
        fill=-1,
    )
    session_id = _session_id(source_path, group)
    source_episode = _attr_text(group.attrs.get("source_episode", episode_name))
    scenarios: list[dict[str, Any]] = []

    keys = np.stack((cube_index, attempt_index), axis=1)
    episode_safe_count = 0
    for segment_start, segment_end in _contiguous_key_runs(keys):
        for core_start, core_end in _find_encounters(
            gap[segment_start:segment_end],
            contact[segment_start:segment_end],
            cfg,
        ):
            core_start += segment_start
            core_end += segment_start
            start = max(segment_start, core_start - int(cfg.margin_frames))
            end = min(segment_end, core_end + int(cfg.margin_frames))
            scenarios.append(
                _scenario_with_source_configuration(
                    group,
                    _scenario_dict(
                        source_path=source_path,
                        session_id=session_id,
                        episode_name=episode_name,
                        source_episode=source_episode,
                        start=start,
                        core_start=core_start,
                        core_end=core_end,
                        end=end,
                        target_severity=_max_severity(
                            gap[core_start:core_end],
                            contact[core_start:core_end],
                            cfg,
                        ),
                        gap=gap,
                        contact=contact,
                        cube_index=cube_index,
                        attempt_index=attempt_index,
                        task_phase=task_phase,
                        controller_event=controller_event,
                        ee_pos=ee_pos,
                        closest_link=closest_link,
                        closest_hand=closest_hand,
                    ),
                )
            )

        if episode_safe_count >= int(cfg.max_safe_per_episode):
            continue
        clear = (
            np.isfinite(gap[segment_start:segment_end])
            & (gap[segment_start:segment_end] > float(cfg.clear_end_m))
            & hand_valid[segment_start:segment_end]
            & ~contact[segment_start:segment_end]
        )
        for local_start, local_end in _true_runs(clear):
            run_start = segment_start + local_start
            run_end = segment_start + local_end
            if run_end - run_start < int(cfg.safe_min_frames):
                continue
            window = min(int(cfg.safe_window_frames), run_end - run_start)
            stride = max(1, int(cfg.safe_stride_frames))
            starts = range(run_start, run_end - window + 1, stride)
            for safe_start in starts:
                if episode_safe_count >= int(cfg.max_safe_per_episode):
                    break
                safe_end = safe_start + window
                scenarios.append(
                    _scenario_with_source_configuration(
                        group,
                        _scenario_dict(
                            source_path=source_path,
                            session_id=session_id,
                            episode_name=episode_name,
                            source_episode=source_episode,
                            start=safe_start,
                            core_start=safe_start,
                            core_end=safe_end,
                            end=safe_end,
                            target_severity="safe",
                            gap=gap,
                            contact=contact,
                            cube_index=cube_index,
                            attempt_index=attempt_index,
                            task_phase=task_phase,
                            controller_event=controller_event,
                            ee_pos=ee_pos,
                            closest_link=closest_link,
                            closest_hand=closest_hand,
                        ),
                    )
                )
                episode_safe_count += 1
            if episode_safe_count >= int(cfg.max_safe_per_episode):
                break
    return scenarios


def _validate_config(cfg: EncounterBuildConfig) -> None:
    if not (
        float(cfg.collision_m)
        <= float(cfg.near_miss_m)
        <= float(cfg.near_m)
        <= float(cfg.gate_start_m)
        < float(cfg.clear_end_m)
    ):
        raise ValueError(
            "Expected collision <= near_miss <= near <= gate_start < clear_end."
        )
    for name in (
        "onset_frames",
        "clear_frames",
        "safe_window_frames",
        "safe_stride_frames",
        "safe_min_frames",
    ):
        if int(getattr(cfg, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    if int(cfg.margin_frames) < 0 or int(cfg.max_safe_per_episode) < 0:
        raise ValueError(
            "margin_frames and max_safe_per_episode must be non-negative."
        )


def _scenario_dict(
    *,
    source_path: str,
    session_id: str,
    episode_name: str,
    source_episode: str,
    start: int,
    core_start: int,
    core_end: int,
    end: int,
    target_severity: str,
    gap: np.ndarray,
    contact: np.ndarray,
    cube_index: np.ndarray,
    attempt_index: np.ndarray,
    task_phase: np.ndarray,
    controller_event: np.ndarray,
    ee_pos: np.ndarray | None,
    closest_link: np.ndarray,
    closest_hand: np.ndarray,
) -> dict[str, Any]:
    core_gap = np.asarray(gap[core_start:core_end], dtype=float)
    finite_core = np.isfinite(core_gap)
    if np.any(finite_core):
        comparable_gap = np.where(finite_core, core_gap, np.inf)
        anchor_step = int(core_start + np.argmin(comparable_gap))
    else:
        anchor_step = int(core_start)
    source_anchor_step = int(start)
    anchor_ee = (
        ee_pos[source_anchor_step].astype(float).tolist()
        if ee_pos is not None
        and np.all(np.isfinite(ee_pos[source_anchor_step]))
        else None
    )
    finite_gap = gap[core_start:core_end]
    finite_gap = finite_gap[np.isfinite(finite_gap)]
    min_gap = float(np.min(finite_gap)) if finite_gap.size else None
    identity = (
        f"{source_path}|{episode_name}|{start}|{core_start}|{core_end}|{end}"
    )
    encounter_id = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    phase_idx = int(task_phase[anchor_step])
    return {
        "id": encounter_id,
        "source_path": os.path.abspath(source_path),
        "session_id": session_id,
        "episode_name": episode_name,
        "source_episode": source_episode,
        "start_step": int(start),
        "core_start_step": int(core_start),
        "core_end_step": int(core_end),
        "end_step": int(end),
        "length": int(end - start),
        "target_severity": target_severity,
        "cube_index": int(cube_index[anchor_step]),
        "attempt_index": int(attempt_index[anchor_step]),
        "task_phase": TASK_PHASE_NAMES[phase_idx],
        "task_phase_index": phase_idx,
        "controller_event": int(controller_event[anchor_step]),
        "phase_anchor_step": anchor_step,
        "trigger_task_phase": TASK_PHASE_NAMES[int(task_phase[source_anchor_step])],
        "trigger_controller_event": int(controller_event[source_anchor_step]),
        "source_anchor_step": source_anchor_step,
        "source_anchor_ee_pos": anchor_ee,
        "recorded_min_surface_gap_m": min_gap,
        "recorded_contact": bool(np.any(contact[core_start:core_end])),
        "recorded_closest_link_id": _mode_int(
            closest_link[core_start:core_end]
        ),
        "recorded_closest_hand_id": _mode_int(
            closest_hand[core_start:core_end]
        ),
    }


def _scenario_with_source_configuration(
    group: h5py.Group,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    scenario["source_configuration"] = extract_episode_source_configuration(
        group,
        active_cube_index=int(scenario["cube_index"]),
        source_anchor_step=int(scenario["source_anchor_step"]),
    )
    return scenario


def _find_encounters(
    gap: np.ndarray,
    contact: np.ndarray,
    cfg: EncounterBuildConfig,
) -> list[tuple[int, int]]:
    gate_active = (gap <= float(cfg.gate_start_m)) | contact
    clear = (gap > float(cfg.clear_end_m)) & ~contact
    encounters: list[tuple[int, int]] = []
    in_encounter = False
    active_count = 0
    clear_count = 0
    core_start = 0
    for index in range(len(gap)):
        if not in_encounter:
            active_count = active_count + 1 if gate_active[index] else 0
            if active_count >= max(1, int(cfg.onset_frames)):
                core_start = index - active_count + 1
                in_encounter = True
                clear_count = 0
            continue
        clear_count = clear_count + 1 if clear[index] else 0
        if clear_count >= max(1, int(cfg.clear_frames)):
            core_end = index - clear_count + 1
            encounters.append((core_start, max(core_start + 1, core_end)))
            in_encounter = False
            active_count = 0
            clear_count = 0
    if in_encounter:
        encounters.append((core_start, len(gap)))
    return encounters


def _max_severity(
    gap: np.ndarray,
    contact: np.ndarray,
    cfg: EncounterBuildConfig,
) -> str:
    if np.any(contact) or np.any(gap <= float(cfg.collision_m)):
        return "collision"
    if np.any(gap <= float(cfg.near_miss_m)):
        return "near_miss"
    if np.any(gap <= float(cfg.near_m)):
        return "near"
    if np.any(gap <= float(cfg.gate_start_m)):
        return "gate_only"
    return "safe"


def _read_scalar(
    group: h5py.Group,
    paths: Iterable[str],
    length: int,
    *,
    fill: float,
) -> np.ndarray:
    for path in paths:
        if path in group:
            values = np.asarray(group[path], dtype=np.float32).reshape(-1)
            return _align_1d(values, length, fill)
    return np.full(length, fill, dtype=np.float32)


def _read_int(
    group: h5py.Group,
    paths: Iterable[str],
    length: int,
    *,
    fill: int,
) -> np.ndarray:
    values = _read_scalar(group, paths, length, fill=float(fill))
    return np.rint(values).astype(np.int32)


def _read_vector(
    group: h5py.Group,
    paths: Iterable[str],
    length: int,
    *,
    width: int,
) -> np.ndarray | None:
    for path in paths:
        if path not in group:
            continue
        values = np.asarray(group[path], dtype=np.float32)
        values = values.reshape(values.shape[0], -1)
        result = np.zeros((length, width), dtype=np.float32)
        count = min(length, values.shape[0])
        result[:count, : min(width, values.shape[1])] = values[
            :count, : min(width, values.shape[1])
        ]
        return result
    return None


def _read_task_phase(
    group: h5py.Group,
    controller_event: np.ndarray,
    length: int,
) -> np.ndarray:
    if "obs/task_phase" in group:
        values = np.asarray(group["obs/task_phase"], dtype=np.float32)
        values = values.reshape(values.shape[0], -1)
        phase = np.argmax(values, axis=1).astype(np.int32)
        return _align_1d(phase, length, 0).astype(np.int32)
    event = np.maximum(controller_event, 0)
    return np.select(
        (event <= 0, event <= 3, event <= 6),
        (0, 1, 2),
        default=3,
    ).astype(np.int32)


def _episode_length(group: h5py.Group) -> int:
    for path in (
        "sim_time",
        "human/head_pos",
        "obs/human_head_pos",
        "obs/ee_pos",
    ):
        if path in group and group[path].shape:
            return int(group[path].shape[0])
    return 0


def _valid_hand_series(
    left: np.ndarray | None,
    right: np.ndarray | None,
    length: int,
) -> np.ndarray:
    result = np.zeros(length, dtype=bool)
    for values in (left, right):
        if values is not None:
            result |= np.all(np.isfinite(values), axis=1) & (
                np.linalg.norm(values, axis=1) > 1e-6
            )
    return result


def _contiguous_key_runs(keys: np.ndarray) -> list[tuple[int, int]]:
    if len(keys) == 0:
        return []
    boundaries = np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(keys)]))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _align_1d(values: np.ndarray, length: int, fill: float) -> np.ndarray:
    result = np.full(length, fill, dtype=np.asarray(values).dtype)
    count = min(length, len(values))
    result[:count] = values[:count]
    return result


def _string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    values = np.asarray(value).reshape(-1)
    return [_attr_text(item) for item in values]


def _finite_array_or_none(value: Any, *, width: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.shape[1] != int(width) or array.shape[0] <= 0:
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def _finite_vector_or_none(
    value: Any,
    *,
    width: int | None = None,
) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if width is not None:
        if array.size != int(width):
            return None
    elif array.size <= 0:
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def _step_vector_or_none(
    group: h5py.Group,
    paths: Sequence[str],
    step: int,
    *,
    width: int | None = None,
) -> np.ndarray | None:
    for path in paths:
        if path not in group:
            continue
        dataset = group[path]
        if not dataset.shape or not 0 <= int(step) < int(dataset.shape[0]):
            continue
        return _finite_vector_or_none(dataset[int(step)], width=width)
    return None


def _complete_panda_joint_positions(
    group: h5py.Group,
    arm_or_full_positions: np.ndarray | None,
    *,
    step: int,
) -> np.ndarray | None:
    if arm_or_full_positions is None:
        return None
    positions = np.asarray(arm_or_full_positions, dtype=np.float64).reshape(-1)
    if positions.size != 7:
        return positions
    gripper_width = _step_vector_or_none(
        group,
        ("obs/gripper_width",),
        step,
        width=1,
    )
    if gripper_width is None:
        return positions
    finger_position = max(0.0, float(gripper_width[0])) / 2.0
    return np.concatenate(
        (positions, np.asarray([finger_position, finger_position], dtype=np.float64))
    )


def _complete_panda_joint_velocities(
    arm_or_full_velocities: np.ndarray | None,
    completed_positions: np.ndarray | None,
) -> np.ndarray | None:
    if arm_or_full_velocities is None:
        return None
    velocities = np.asarray(arm_or_full_velocities, dtype=np.float64).reshape(-1)
    if (
        velocities.size == 7
        and completed_positions is not None
        and np.asarray(completed_positions).size == 9
    ):
        return np.concatenate((velocities, np.zeros(2, dtype=np.float64)))
    return velocities


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        scalar = np.asarray(value).reshape(-1)[0]
        result = int(scalar)
    except (IndexError, TypeError, ValueError, OverflowError):
        return None
    return result


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    array = np.asarray(value).reshape(-1)
    if array.size <= 0:
        return None
    text = _attr_text(array[0]).strip()
    return text or None


def _array_list(value: np.ndarray | None) -> list[Any] | None:
    if value is None:
        return None
    return np.asarray(value).tolist()


def _session_id(source_path: str, group: h5py.Group) -> str:
    for key in ("source_session", "session_id", "source_file"):
        if key in group.attrs:
            text = _attr_text(group.attrs[key])
            if text:
                return os.path.splitext(os.path.basename(text))[0]
    return os.path.splitext(os.path.basename(source_path))[0]


def _attr_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _mode_int(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.int64)
    values = values[values >= 0]
    if values.size == 0:
        return -1
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])
