#!/usr/bin/env python3
"""Collect frozen-BC/fixed-CBF single-pick VR explicit-feedback trials."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for candidate in (PROJECT_ROOT, SCRIPT_DIR, SCRIPT_DIR / "rl"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from v3_chan.collection_provenance import (  # noqa: E402
    resolve_verified_code_provenance,
)
from v3_chan.explicit_feedback.action_trace import ActionTraceRecorder  # noqa: E402
from v3_chan.explicit_feedback.online_config import (  # noqa: E402
    DEFAULT_PROTOCOL_CONFIG,
    DEFAULT_SCENARIO_CONFIG,
    actual_cbf_config,
    canonical_sha256,
    load_json_object,
    validate_frozen_protocol_config,
)
from v3_chan.explicit_feedback.online_protocol import (  # noqa: E402
    CrossingPathMonitor,
    EncounterDetector,
    EncounterDetectorConfig,
    TrialSpec,
    build_participant_schedule,
)
from v3_chan.explicit_feedback.online_recorder import OnlineExplicitFeedbackRecorder  # noqa: E402
from v3_chan.explicit_feedback.online_rows import build_online_transition_row  # noqa: E402
from v3_chan.explicit_feedback.online_scenario import (  # noqa: E402
    NOMINAL_BC_SWEEP_SEMANTICS,
    CrossingCorridor,
    CrossingCueVisuals,
    NominalBCCorridorSweep,
    ProximalArmProtocolMonitor,
    calibrate_surface_gap_corridor,
)
from v3_chan.explicit_feedback.online_schema import (  # noqa: E402
    EncounterRecordV1,
    Q1_ID,
    QueryRecord,
    RealtimeMarkerRecord,
    TrialPlan,
)
from v3_chan.explicit_feedback.policy import FrozenBCPolicy  # noqa: E402
from v3_chan.explicit_feedback.rows import validate_runtime_step  # noqa: E402
from v3_chan.explicit_feedback.runtime import (  # noqa: E402
    apply_controlled_hold,
    measured_joint_state,
    validate_environment_contract,
)
from v3_chan.explicit_feedback.schema import (  # noqa: E402
    POLICY_RELATIVE_PATH,
    POLICY_SHA256,
    RUNTIME_CONTRACT_SHA256,
    RUNTIME_HANDOFF_COMMIT,
)
from v3_chan.explicit_feedback.tracking import TrackingDropout, TrackingWatchdog  # noqa: E402
from v3_chan.explicit_feedback.tracking_guard import PreApplyTrackingGuard  # noqa: E402
from v3_chan.runtime_contracts.bc_cbf_explicit_feedback_v1.validate_runtime_contract import (  # noqa: E402
    validate_runtime_contract,
)


class CollectionAbort(RuntimeError):
    pass


@dataclass
class LoggedStep:
    obs: np.ndarray
    info: dict[str, Any]
    tracking: Any
    terminated: bool
    truncated: bool
    source_step: int
    result_step: int
    simulation_time_s: float
    monotonic_ns: int
    unix_ns: int
    minimum_gap_m: float
    left_gap_m: float
    right_gap_m: float
    ttc_s: float
    closing_speed_m_s: float
    dynamic_measurement_valid: bool
    cbf_active: bool
    intervention_norm_rad_s: float
    crossing_update: Any = None
    encounter_update: Any = None
    questionnaire_result: Any = None
    realtime_markers: tuple[Any, ...] = ()
    proximal_gap_m: float = 10.0
    proximal_collider_path: str = ""
    left_proximal_gap_m: float = 10.0
    right_proximal_gap_m: float = 10.0
    left_proximal_collider_path: str = ""
    right_proximal_collider_path: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL_CONFIG)
    parser.add_argument("--scenario-config", type=Path, default=DEFAULT_SCENARIO_CONFIG)
    parser.add_argument("--practice-trials", type=int, default=4)
    parser.add_argument(
        "--include-anchor-repeats",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: on for full production, off for pilot.",
    )
    parser.add_argument(
        "--pilot-evaluated-trials",
        type=int,
        default=None,
        help="Pilot mode with 12-16 total evaluated trials.",
    )
    parser.add_argument(
        "--practice-only",
        action="store_true",
        help="Run only practice trials; intended for equipment smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the frozen schedule without starting Isaac Sim.",
    )
    parser.add_argument("--reset-after-query", action="store_true")
    parser.add_argument("--tracking-timeout-s", type=float, default=120.0)
    parser.add_argument("--max-pose-age-ms", type=float, default=100.0)
    parser.add_argument("--tracking-stable-frames", type=int, default=6)
    parser.add_argument(
        "--code-commit-sha",
        default=os.environ.get("HRI_CODE_COMMIT_SHA", ""),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not str(args.participant_id).strip() or not str(args.session_id).strip():
        raise ValueError("participant/session IDs must be non-empty pseudonyms")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.practice_trials < 0:
        raise ValueError("--practice-trials must be non-negative")
    if args.practice_only and args.practice_trials <= 0:
        raise ValueError("--practice-only requires at least one practice trial")
    if args.practice_only and args.practice_trials > 4:
        raise ValueError("--practice-only supports one to four equipment trials")
    if not args.practice_only and args.practice_trials != 4:
        raise ValueError("pilot/production protocol requires exactly four practice trials")
    if args.pilot_evaluated_trials is not None and not 12 <= args.pilot_evaluated_trials <= 16:
        raise ValueError("--pilot-evaluated-trials must be in [12, 16]")
    if args.tracking_stable_frames < 1:
        raise ValueError("--tracking-stable-frames must be positive")
    if not math.isfinite(args.tracking_timeout_s) or args.tracking_timeout_s <= 0:
        raise ValueError("--tracking-timeout-s must be finite and positive")
    if not math.isfinite(args.max_pose_age_ms) or args.max_pose_age_ms <= 0:
        raise ValueError("--max-pose-age-ms must be finite and positive")
    commit = str(args.code_commit_sha).strip().lower()
    if commit and (
        len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise ValueError(
            "--code-commit-sha, when supplied, must be a 40-character lowercase Git SHA"
        )


def _load_configs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_json_object(args.protocol_config)
    scenario = load_json_object(args.scenario_config)
    validate_frozen_protocol_config(protocol, scenario)
    return protocol, scenario


def _build_schedule(
    args: argparse.Namespace,
) -> tuple[tuple[TrialSpec, ...], tuple[TrialPlan, ...]]:
    anchors = args.include_anchor_repeats
    if anchors is None:
        anchors = args.pilot_evaluated_trials is None
    schedule = build_participant_schedule(
        str(args.participant_id),
        session_id=str(args.session_id),
        seed=int(args.seed),
        practice_trials=int(args.practice_trials),
        include_anchor_repeats=bool(anchors),
        pilot_evaluated_trials=args.pilot_evaluated_trials,
    )
    specs = tuple(schedule.practice_trials if args.practice_only else schedule.trials)
    plans = tuple(_trial_plan(spec, index=index) for index, spec in enumerate(specs))
    return specs, plans


def _trial_plan(spec: TrialSpec, *, index: int) -> TrialPlan:
    crossing_hand = "left" if spec.crossing_direction == "left_to_right" else "right"
    return TrialPlan(
        trial_index=index,
        trial_id=spec.trial_id,
        query_id=spec.query_id,
        task_phase=spec.task_phase,
        severity=spec.severity,
        direction=spec.crossing_direction,
        speed=spec.crossing_speed,
        crossing_hand=crossing_hand,
        practice=spec.practice,
        anchor_repeat=spec.anchor_repeat,
        pilot=spec.pilot,
        analysis_exclude=spec.analysis_exclude,
        source_trial_id=spec.repeated_trial_id or "",
    )


def _default_output(args: argparse.Namespace) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(args.session_id)
    )
    return (
        PROJECT_ROOT
        / "v3_chan"
        / "explicit_feedback_data"
        / f"{session}_{timestamp}.hdf5"
    )


def _simulation_config() -> dict[str, Any]:
    isaac_root = Path(
        os.environ.get("ISAACSIM_ROOT", str(Path.home() / "isaac-sim-4.5.0"))
    )
    xr_mode = os.environ.get("ISAAC_XR_MODE", "vr").strip().lower()
    name = (
        "isaacsim.exp.base.xr.openxr.kit"
        if xr_mode == "openxr"
        else "isaacsim.exp.base.xr.vr.kit"
    )
    experience = isaac_root / "apps" / name
    if not experience.is_file():
        raise FileNotFoundError(f"Live VR experience not found: {experience}")
    return {
        "headless": False,
        "width": 1280,
        "height": 720,
        "active_gpu": 0,
        "physics_gpu": 0,
        "multi_gpu": False,
        "max_gpu_count": 1,
        "experience": str(experience),
    }


def _build_environment(provider: Any, *, seed: int):
    from v3_chan.rl.pick_place_env import IsaacPickPlaceEnv, PickPlaceEnvConfig

    return IsaacPickPlaceEnv(
        PickPlaceEnvConfig(
            cube_count=1,
            max_episode_steps=12_000,
            success_dist=0.06,
            action_scale=1.0,
            action_version="action_v1_controller_target_delta",
            fixed_orientation=True,
            gripper_mode="event",
            require_release_for_success=True,
            strict_task_semantics=True,
            strict_place_xy_tolerance_m=0.05,
            reward_version="reward_v4_post_release_stability_hri_errp",
            observation_mode="flat",
            seed=int(seed),
            render=True,
            pseudo_errp_enabled=False,
            pseudo_errp_sources=(),
            visualize_human_replay=True,
            synthetic_human_enabled=False,
            physical_safety_controller="cbf",
            cbf_safe_gap_m=0.05,
            cbf_activation_gap_m=0.13,
            cbf_gamma_per_s=8.0,
            cbf_prediction_horizon_s=0.15,
            cbf_max_prediction_buffer_m=0.08,
            cbf_max_joint_speed_rad_s=2.0,
            extended_backup_safety_geometry=False,
        ),
        human_state_fn=provider,
    )


def _capture_scene_and_restoration(
    env: Any, *, seed: int
) -> tuple[dict[str, Any], dict[str, Any], str]:
    from v3_chan.scene_randomization import scene_layout_id

    cube_poses = [cube.get_world_pose() for cube in env.cubes]
    target_position, target_orientation = env.place_target.get_world_pose()
    positions, velocities = measured_joint_state(
        env.robot, joint_count=len(env.robot.dof_names)
    )
    names = tuple(str(cube.name) for cube in env.cubes)
    cube_positions = np.asarray([pose[0] for pose in cube_poses], dtype=np.float64)
    cube_orientations = np.asarray([pose[1] for pose in cube_poses], dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    target_orientation = np.asarray(target_orientation, dtype=np.float64)
    scene = {
        "cube_names": names,
        "cube_positions_world_m": cube_positions,
        "cube_orientations_wxyz": cube_orientations,
        "place_target_position_world_m": target_position,
        "place_target_orientation_wxyz": target_orientation,
        "robot_initial_joint_positions_rad": positions,
        "robot_initial_joint_velocities_rad_s": velocities,
    }
    source_configuration = {
        "cube_names": names,
        "cube_positions_world": cube_positions,
        "cube_orientations_wxyz": cube_orientations,
        "place_target_position_world": target_position,
        "place_target_orientation_wxyz": target_orientation,
        "robot_initial_joint_positions": positions,
        "robot_initial_joint_velocities": velocities,
    }
    restoration = {
        "source_configuration_available": True,
        "restoration_mode": "exact_pose",
        "restoration_reason": "passed_frozen_bc_only_precheck",
        "source_cube_index": 0,
        "source_cube_name": names[0],
        "collection_seed": int(seed),
        "layout_seed": int(seed),
        "source_configuration": source_configuration,
    }
    layout_id = scene_layout_id(
        cube_positions, cube_orientations, target_position, target_orientation
    )
    return scene, restoration, layout_id


def _strict_state(info: Mapping[str, Any]) -> dict[str, Any]:
    value = info.get("strict_task_semantics", {})
    value = dict(value) if isinstance(value, Mapping) else {}
    state = value.get("state", {})
    return dict(state) if isinstance(state, Mapping) else {}


def _wait_for_tracking_and_clear(
    *,
    env: Any,
    provider: Any,
    simulation_app: Any,
    joint_count: int,
    timeout_s: float,
    stable_frames: int,
    clear_gap_m: float,
    proximal_monitor: ProximalArmProtocolMonitor | None,
    stop_requested: Callable[[], None],
) -> tuple[np.ndarray, dict[str, Any], Any]:
    """Refresh a paused initial observation until tracking and both gaps are clear."""

    env.world.pause()
    apply_controlled_hold(env.robot, joint_count=joint_count)
    started = time.monotonic()
    stable = 0
    last_reason = "no observation"
    while stable < stable_frames:
        stop_requested()
        obs, info = env.refresh_observation()
        snapshot = provider.observation_snapshot
        try:
            if snapshot is None:
                raise RuntimeError("missing observation tracking snapshot")
            provider.watchdog.require_current(snapshot)
            proximal_clear = True
            if proximal_monitor is not None:
                proximal_clear = all(
                    proximal_monitor.closest_surface_gap_m(
                        getattr(snapshot, hand).position_world
                    )[0]
                    > float(clear_gap_m)
                    for hand in ("left", "right")
                )
            gaps_clear = (
                bool(info.get("geometry_valid", False))
                and float(info.get("left_end_effector_surface_gap_m", -1.0))
                > float(clear_gap_m)
                and float(info.get("right_end_effector_surface_gap_m", -1.0))
                > float(clear_gap_m)
                and proximal_clear
            )
            if not gaps_clear:
                raise RuntimeError(
                    "move both hands to staging "
                    f"(>{float(clear_gap_m):.3f} m protected-surface gap)"
                )
        except (TrackingDropout, RuntimeError) as error:
            stable = 0
            last_reason = str(error)
        else:
            stable += 1
            last_reason = ""
        apply_controlled_hold(env.robot, joint_count=joint_count)
        simulation_app.update()
        if time.monotonic() - started > timeout_s:
            raise CollectionAbort("tracking/clear staging timeout: " + last_reason)
        time.sleep(0.01)
    env.world.play()
    assert snapshot is not None
    return np.asarray(obs, dtype=np.float32), dict(info), snapshot


def _feedback_clocks(env: Any, *, monotonic_ns: int, unix_ns: int):
    from v3_chan.explicit_feedback.xr_feedback import FeedbackClocks

    return FeedbackClocks(
        sim_time_s=max(0.0, float(getattr(env.world, "current_time", 0.0))),
        monotonic_ns=int(monotonic_ns),
        unix_ns=int(unix_ns),
        control_step=max(0, int(getattr(env, "step_count", 0))),
    )


def _wait_for_controller_release(
    *,
    source: Any,
    feedback: Any,
    provider: Any,
    simulation_app: Any,
    env: Any,
    timeout_s: float,
    stop_requested: Callable[[], None],
) -> None:
    started = time.monotonic()
    stable = 0
    feedback.set_crossing_hand(None)
    while stable < 3:
        stop_requested()
        # Refresh VRAvatar's per-frame device cache before reading buttons.
        # The button adapter then reuses those wrappers instead of querying
        # XRCore a second time during the same frame.
        provider.sample()
        snapshot = source.read()
        known_release = bool(
            snapshot.left_connected is True
            and snapshot.right_connected is True
            and snapshot.left_x is False
            and snapshot.left_y is False
            and snapshot.right_a is False
            and snapshot.right_b is False
        )
        stable = stable + 1 if known_release else 0
        feedback.update(
            snapshot,
            _feedback_clocks(
                env, monotonic_ns=time.monotonic_ns(), unix_ns=time.time_ns()
            ),
        )
        simulation_app.update()
        if time.monotonic() - started > timeout_s:
            detail = getattr(source, "last_error", "") or "buttons unavailable/held"
            raise CollectionAbort("VR controller preflight failed: " + detail)
        time.sleep(0.01)


def _wait_for_clear_control_observation(
    *,
    env: Any,
    provider: Any,
    simulation_app: Any,
    joint_count: int,
    timeout_s: float,
    clear_confirmation_frames: int,
    stop_requested: Callable[[], None],
    context: str,
) -> tuple[np.ndarray, dict[str, Any], Any]:
    """Legacy collector helper retained for API/schema compatibility."""

    apply_controlled_hold(env.robot, joint_count=joint_count)
    env.world.pause()
    required_clear = max(1, int(clear_confirmation_frames))
    clear_frames = 0
    started = time.monotonic()
    last_reason = "no control observation"
    while clear_frames < required_clear:
        stop_requested()
        obs, info = env.refresh_observation()
        snapshot = provider.observation_snapshot
        tracking_error = ""
        if snapshot is None:
            tracking_error = "missing tracking snapshot"
        else:
            try:
                provider.watchdog.require_current(snapshot)
            except TrackingDropout as error:
                tracking_error = str(error)
        geometry_valid = bool(info.get("geometry_valid", False))
        left_gap = float(info.get("left_end_effector_surface_gap_m", math.nan))
        right_gap = float(info.get("right_end_effector_surface_gap_m", math.nan))
        clear = bool(
            not tracking_error
            and geometry_valid
            and math.isfinite(left_gap)
            and math.isfinite(right_gap)
            and left_gap > 0.13
            and right_gap > 0.13
        )
        if clear:
            clear_frames += 1
            last_reason = ""
        else:
            clear_frames = 0
            last_reason = (
                f"tracking={tracking_error or 'valid'}, "
                f"geometry_valid={geometry_valid}, left_gap={left_gap:.6g}, "
                f"right_gap={right_gap:.6g}"
            )
        if clear_frames >= required_clear:
            assert snapshot is not None
            env.world.play()
            return np.asarray(obs, dtype=np.float32), dict(info), snapshot
        apply_controlled_hold(env.robot, joint_count=joint_count)
        is_playing = getattr(env.world, "is_playing", None)
        if callable(is_playing) and bool(is_playing()):
            raise CollectionAbort(
                f"{context} readiness attempted while physics was playing"
            )
        simulation_app.update()
        if time.monotonic() - started >= timeout_s:
            raise CollectionAbort(
                f"{context} did not reach a stable clear CBF band: "
                + last_reason
            )
        time.sleep(0.01)
    raise AssertionError("unreachable clear-control-observation state")


def _append_unrecorded_responses(
    *,
    broker: Any,
    recorder: Any,
    recorded_response_ids: set[str],
) -> None:
    """Drain each legacy server response audit record exactly once."""

    for response in broker.snapshot()[1]:
        response_id = str(response.response_id)
        if response_id in recorded_response_ids:
            continue
        recorder.append_response(response)
        recorded_response_ids.add(response_id)


def _record_prompt_exchange(
    *,
    env: Any,
    provider: Any,
    broker: Any,
    recorder: Any,
    encounter: Any,
    joint_count: int,
    tracking_timeout_s: float,
    tracking_stable_frames: int,
    clear_confirmation_frames: int,
    simulation_app: Any,
    stop_requested: Callable[[], None],
    abort_reason_fn: Callable[[], str],
    resume_after_prompt: bool,
    recorded_response_ids: set[str],
) -> tuple[np.ndarray, dict[str, Any], Any] | None:
    """Retain the prior browser-prompt helper without using it in v1 online trials."""

    from v3_chan.explicit_feedback.protocol import FeedbackTimeout

    apply_controlled_hold(env.robot, joint_count=joint_count)
    env.world.pause()
    prompt = broker.publish_prompt(encounter.encounter_id)

    def pump_paused_xr_frame() -> None:
        stop_requested()
        is_playing = getattr(env.world, "is_playing", None)
        if callable(is_playing) and bool(is_playing()):
            raise CollectionAbort(
                "explicit-feedback wait attempted while physics was playing"
            )
        simulation_app.update()
        stop_requested()

    try:
        broker.wait_for_response(
            prompt.prompt_id,
            idle_callback=pump_paused_xr_frame,
            poll_interval_s=0.01,
        )
    except (FeedbackTimeout, CollectionAbort) as error:
        stop_reason = str(abort_reason_fn() or "").strip()
        if isinstance(error, CollectionAbort):
            stop_reason = stop_reason or str(error)
        if stop_reason:
            broker.close()
        final_prompt = broker.prompt_record(prompt.prompt_id)
        encounter = replace(
            encounter,
            prompt_id=prompt.prompt_id,
            terminal_status=(
                "collection_aborted" if stop_reason else "response_timeout"
            ),
            abort_reason=stop_reason or "explicit_feedback_timeout",
        )
        recorder.append_encounter(encounter)
        recorder.append_prompt(final_prompt)
        _append_unrecorded_responses(
            broker=broker,
            recorder=recorder,
            recorded_response_ids=recorded_response_ids,
        )
        raise CollectionAbort(stop_reason or str(error)) from error
    final_prompt = broker.prompt_record(prompt.prompt_id)
    recorder.append_encounter(replace(encounter, prompt_id=prompt.prompt_id))
    recorder.append_prompt(final_prompt)
    _append_unrecorded_responses(
        broker=broker,
        recorder=recorder,
        recorded_response_ids=recorded_response_ids,
    )
    if not resume_after_prompt:
        return None
    stop_requested()
    provider.wait_until_ready(
        simulation_app,
        timeout_s=tracking_timeout_s,
        stable_frames=tracking_stable_frames,
        stop_requested=stop_requested,
    )
    return _wait_for_clear_control_observation(
        env=env,
        provider=provider,
        simulation_app=simulation_app,
        joint_count=joint_count,
        timeout_s=tracking_timeout_s,
        clear_confirmation_frames=clear_confirmation_frames,
        stop_requested=stop_requested,
        context="post-feedback resume",
    )


def _commit_validated_with_signal_barrier(
    *,
    recorder: Any,
    abort_reason_fn: Callable[[], str],
    commit_state: dict[str, bool],
) -> Path:
    """Make the validated rename an explicit SIGINT/SIGTERM boundary."""

    required_signal_api = (
        "pthread_sigmask",
        "sigpending",
        "SIG_BLOCK",
        "SIG_SETMASK",
    )
    missing = [name for name in required_signal_api if not hasattr(signal, name)]
    if missing:
        raise RuntimeError(
            "atomic collection commit requires POSIX signal masking; missing "
            + ", ".join(missing)
        )
    watched = {signal.SIGINT, signal.SIGTERM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    final_path: Path | None = None

    def pending_reason() -> str:
        pending = set(signal.sigpending()).intersection(watched)
        if not pending:
            return ""
        return f"signal_{min(int(signum) for signum in pending)}"

    def restore_partial(reason: str) -> None:
        commit_state["value"] = False
        assert final_path is not None
        os.replace(final_path, recorder.partial_path)
        raise CollectionAbort(reason)

    try:
        reason = str(abort_reason_fn() or "").strip() or pending_reason()
        if reason:
            raise CollectionAbort(reason)
        final_path = recorder.commit_validated()
        reason = str(abort_reason_fn() or "").strip() or pending_reason()
        if reason:
            restore_partial(reason)
        commit_state["value"] = True
        reason = str(abort_reason_fn() or "").strip() or pending_reason()
        if reason:
            restore_partial(reason)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    # Unmasking may synchronously deliver a signal to request_stop().  Check
    # once more after restoration so a just-committed artifact is moved back
    # to its recoverable .partial name rather than being published.
    reason = str(abort_reason_fn() or "").strip()
    if reason:
        restore_partial(reason)
    assert final_path is not None
    return final_path


def _nominal_sweep_offsets(config: Mapping[str, Any]) -> tuple[float, ...]:
    bounds = tuple(
        float(value) for value in config["candidate_outward_offset_range_m"]
    )
    if len(bounds) != 2:
        raise ValueError("candidate_outward_offset_range_m must have two values")
    low, high = bounds
    step = float(config["candidate_outward_offset_step_m"])
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError("nominal sweep offset range is invalid")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("nominal sweep offset step must be positive")
    count_float = (high - low) / step
    count = int(round(count_float))
    if not math.isclose(count_float, count, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("nominal sweep range must be divisible by its step")
    return tuple(float(low + index * step) for index in range(count + 1))


def _require_clear_nominal_calibration_state(
    *,
    info: Mapping[str, Any],
    provider: Any,
    proximal_monitor: ProximalArmProtocolMonitor,
    clear_gap_m: float,
) -> Any:
    snapshot = provider.observation_snapshot
    if snapshot is None:
        raise CollectionAbort("nominal corridor sweep has no tracking snapshot")
    provider.watchdog.require_current(snapshot)
    distal = (
        float(info.get("left_end_effector_surface_gap_m", math.nan)),
        float(info.get("right_end_effector_surface_gap_m", math.nan)),
    )
    proximal = tuple(
        proximal_monitor.closest_surface_gap_m(
            getattr(snapshot, hand).position_world
        )[0]
        for hand in ("left", "right")
    )
    if (
        not bool(info.get("geometry_valid", False))
        or not all(math.isfinite(value) for value in (*distal, *proximal))
        or min(*distal, *proximal) < float(clear_gap_m)
    ):
        raise CollectionAbort(
            "hands entered protected/proximal space during BC-only nominal "
            "corridor calibration"
        )
    return snapshot


def _nominal_trajectory_sample(
    *,
    env: Any,
    info: Mapping[str, Any],
    phase_start_simulation_time_s: float,
    arm_joint_indices: Sequence[int],
) -> dict[str, Any]:
    positions, velocities = measured_joint_state(
        env.robot, joint_count=len(env.robot.dof_names)
    )
    indices = np.asarray(arm_joint_indices, dtype=int)
    ee_position, ee_orientation = _ee_pose(env)
    simulation_time_s = float(
        info.get("sim_time", getattr(env.world, "current_time", 0.0))
    )
    return {
        "relative_time_s": simulation_time_s
        - float(phase_start_simulation_time_s),
        "simulation_time_s": simulation_time_s,
        "control_step": int(info.get("step", 0)),
        "controller_event": int(info.get("controller_event", -1)),
        "ee_position_world_m": ee_position.tolist(),
        "ee_orientation_wxyz": ee_orientation.tolist(),
        "arm_joint_positions_rad": np.asarray(positions, dtype=float)[
            indices
        ].tolist(),
        "arm_joint_velocities_rad_s": np.asarray(velocities, dtype=float)[
            indices
        ].tolist(),
    }


def _calibrate_nominal_corridor_bank(
    *,
    env: Any,
    policy: FrozenBCPolicy,
    provider: Any,
    protocol: Mapping[str, Any],
    scenario: Mapping[str, Any],
    restoration: Mapping[str, Any],
    seed: int,
    simulation_app: Any,
    proximal_monitor: ProximalArmProtocolMonitor,
    arm_joint_indices: Sequence[int],
    stop_requested: Callable[[], None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build severity corridors from replayed, time-aligned BC-only motion.

    Each phase is replayed from the accepted source layout.  At its trigger the
    task-event clock is held only through the cue, then advances normally from
    START through crossing/recovery while direct frozen BC drives RMPFlow with
    CBF disabled.  The sweep queries hypothetical hand paths against the moving
    protected PhysX geometry; actual tracked-hand gaps never define severity.
    """

    reference_cfg = dict(scenario["nominal_bc_reference"])
    crossing_cfg = dict(scenario["crossing"])
    layout_cfg = dict(scenario["layout_precheck"])
    offsets = _nominal_sweep_offsets(reference_cfg)
    sample_interval_s = float(reference_cfg["sweep_sample_interval_s"])
    maximum_sample_interval_s = float(
        reference_cfg["maximum_sweep_sample_interval_s"]
    )
    minimum_samples = int(reference_cfg["minimum_trajectory_samples"])
    if not (
        0.0 < sample_interval_s <= maximum_sample_interval_s
        and minimum_samples >= 2
    ):
        raise ValueError("invalid nominal trajectory sampling contract")
    trigger_events = {
        str(phase): int(event)
        for phase, event in protocol["task"]["crossing_trigger_event"].items()
    }
    saved_references = restoration.get("nominal_bc_phase_references", {})
    if not isinstance(saved_references, Mapping):
        raise RuntimeError("nominal phase references are unavailable")

    trajectories: dict[str, Any] = {}
    bank: dict[str, Any] = {}
    base_severity = dict(scenario["severity"]["shallow"])
    maximum_steps = int(layout_cfg["maximum_steps"])
    clear_gap_m = float(layout_cfg["hands_clear_gap_m"])

    for phase, trigger_event in trigger_events.items():
        stop_requested()
        obs, info = env.reset(
            seed=int(seed), source_restoration=dict(restoration)
        )
        obs, info, _tracking = _wait_for_tracking_and_clear(
            env=env,
            provider=provider,
            simulation_app=simulation_app,
            joint_count=len(env.robot.dof_names),
            timeout_s=60.0,
            stable_frames=int(layout_cfg["hands_clear_confirmation_frames"]),
            clear_gap_m=clear_gap_m,
            proximal_monitor=proximal_monitor,
            stop_requested=stop_requested,
        )
        reached = False
        for _ in range(maximum_steps):
            _require_clear_nominal_calibration_state(
                info=info,
                provider=provider,
                proximal_monitor=proximal_monitor,
                clear_gap_m=clear_gap_m,
            )
            if int(info.get("controller_event", -1)) == trigger_event:
                reached = True
                break
            action = policy.predict(np.asarray(obs, dtype=np.float32)).action
            obs, _reward, terminated, truncated, info = env.step(
                action, advance_task_phase=True
            )
            obs = np.asarray(obs, dtype=np.float32)
            if (terminated or truncated) and not bool(info.get("success", False)):
                break
        if not reached:
            raise RuntimeError(
                f"BC-only nominal calibration did not reach {phase} event "
                f"{trigger_event}"
            )

        trigger_reference = saved_references.get(phase, {})
        if not isinstance(trigger_reference, Mapping):
            raise RuntimeError(f"missing accepted trigger reference for {phase}")
        trigger_sample = _nominal_trajectory_sample(
            env=env,
            info=info,
            phase_start_simulation_time_s=float(
                info.get("sim_time", getattr(env.world, "current_time", 0.0))
            ),
            arm_joint_indices=arm_joint_indices,
        )
        reference_ee = np.asarray(
            trigger_reference.get("ee_position_world_m", ()), dtype=float
        ).reshape(-1)
        reference_joints = np.asarray(
            trigger_reference.get("arm_joint_positions_rad", ()), dtype=float
        ).reshape(-1)
        trigger_ee = np.asarray(
            trigger_sample["ee_position_world_m"], dtype=float
        )
        trigger_joints = np.asarray(
            trigger_sample["arm_joint_positions_rad"], dtype=float
        )
        if reference_ee.shape != (3,) or reference_joints.shape != (
            len(arm_joint_indices),
        ):
            raise RuntimeError(f"invalid accepted trigger reference for {phase}")
        if (
            float(np.linalg.norm(trigger_ee - reference_ee))
            > float(reference_cfg["maximum_ee_position_error_m"])
            or float(np.max(np.abs(trigger_joints - reference_joints)))
            > float(reference_cfg["maximum_arm_joint_error_rad"])
        ):
            raise RuntimeError(
                f"BC-only nominal calibration trigger drifted for {phase}"
            )

        phase_start = float(trigger_sample["simulation_time_s"])
        base_corridor = calibrate_surface_gap_corridor(
            env.safety_geometry,
            hand="left",
            end_effector_position_world_m=trigger_ee,
            target_gap_m=float(base_severity["target_gap_m"]),
            allowed_gap_range_m=base_severity["minimum_gap_range_m"],
            corridor_length_m=float(crossing_cfg["corridor_length_m"]),
        )
        sweeps: dict[tuple[str, str], NominalBCCorridorSweep] = {}
        for direction in ("left_to_right", "right_to_left"):
            hand = "left" if direction == "left_to_right" else "right"
            for speed_name in ("slow", "fast"):
                sweeps[(direction, speed_name)] = NominalBCCorridorSweep(
                    phase=phase,
                    base_corridor=base_corridor,
                    crossing_direction=direction,
                    speed_m_s=float(
                        crossing_cfg["speed_target_m_s"][speed_name]
                    ),
                    cue_lead_s=float(crossing_cfg["cue_lead_s"]),
                    corridor_length_m=float(crossing_cfg["corridor_length_m"]),
                    hand=hand,
                    candidate_normal_offsets_m=offsets,
                    phase_start_simulation_time_s=phase_start,
                    maximum_sample_interval_s=maximum_sample_interval_s,
                )
        maximum_end = max(
            sweep.crossing_end_simulation_time_s for sweep in sweeps.values()
        )
        crossing_start_time = min(
            sweep.crossing_start_simulation_time_s for sweep in sweeps.values()
        )
        samples: list[dict[str, Any]] = []
        last_sample_time = -math.inf
        calibration_steps = 0
        while True:
            stop_requested()
            _require_clear_nominal_calibration_state(
                info=info,
                provider=provider,
                proximal_monitor=proximal_monitor,
                clear_gap_m=clear_gap_m,
            )
            now = float(
                info.get("sim_time", getattr(env.world, "current_time", 0.0))
            )
            should_sample = bool(
                not samples
                or now - last_sample_time >= sample_interval_s - 1e-12
                or now >= maximum_end
            )
            if should_sample:
                samples.append(
                    _nominal_trajectory_sample(
                        env=env,
                        info=info,
                        phase_start_simulation_time_s=phase_start,
                        arm_joint_indices=arm_joint_indices,
                    )
                )
                for sweep in sweeps.values():
                    sweep.observe(
                        simulation_time_s=now,
                        safety_geometry=env.safety_geometry,
                    )
                last_sample_time = now
            if now >= maximum_end:
                break
            if calibration_steps >= maximum_steps:
                raise RuntimeError(
                    f"BC-only nominal calibration horizon exhausted for {phase}"
                )
            action = policy.predict(np.asarray(obs, dtype=np.float32)).action
            obs, _reward, _terminated, _truncated, info = env.step(
                action,
                advance_task_phase=bool(now >= crossing_start_time),
            )
            obs = np.asarray(obs, dtype=np.float32)
            calibration_steps += 1
            if (
                now < crossing_start_time
                and int(info.get("controller_event", -1)) != trigger_event
            ):
                raise RuntimeError(
                    f"cue-latched BC-only nominal calibration left {phase} event"
                )
            strict = _strict_state(info)
            if str(strict.get("failure_reason", "")):
                raise RuntimeError(
                    f"BC-only nominal calibration failed for {phase}: "
                    + str(strict["failure_reason"])
                )
        if len(samples) < minimum_samples:
            raise RuntimeError(
                f"BC-only nominal trajectory for {phase} has only "
                f"{len(samples)} samples"
            )
        trajectory_payload: dict[str, Any] = {
            "semantics": NOMINAL_BC_SWEEP_SEMANTICS,
            "phase": phase,
            "controller": "bc_only",
            "cbf_enabled": False,
            "cue_phase_clock_latched": True,
            "task_progression_during_crossing": True,
            "trigger_event": trigger_event,
            "sample_interval_target_s": sample_interval_s,
            "maximum_sample_interval_s": maximum_sample_interval_s,
            "samples": samples,
        }
        trajectory_payload["sha256"] = canonical_sha256(
            {key: value for key, value in trajectory_payload.items() if key != "sha256"}
        )
        trajectories[phase] = trajectory_payload

        phase_bank: dict[str, Any] = {}
        for (direction, speed_name), sweep in sweeps.items():
            direction_bank = phase_bank.setdefault(direction, {})
            speed_bank: dict[str, Any] = {}
            for severity_name in ("control", "shallow", "threat"):
                severity_cfg = dict(scenario["severity"][severity_name])
                audit = sweep.select_severity(
                    severity=severity_name,
                    target_gap_m=float(severity_cfg["target_gap_m"]),
                    allowed_gap_range_m=severity_cfg["minimum_gap_range_m"],
                    minimum_pre_cue_start_gap_m=float(
                        reference_cfg["minimum_pre_cue_start_gap_m"]
                    ),
                ).as_dict()
                audit["trajectory_sha256"] = trajectory_payload["sha256"]
                audit["trajectory_sample_count"] = len(samples)
                audit["trigger_reference_control_step"] = int(
                    trigger_reference.get("control_step", -1)
                )
                audit["trigger_reference_simulation_time_s"] = float(
                    trigger_reference.get("simulation_time_s", -1.0)
                )
                audit["sweep_audit_sha256"] = canonical_sha256(
                    {"nominal_bc_sweep": audit}
                )
                speed_bank[severity_name] = audit
            direction_bank[speed_name] = speed_bank
        bank[phase] = phase_bank
        print(
            f"[OnlineFeedback][NominalSweep] phase={phase} "
            f"samples={len(samples)} conditions=12 "
            f"trajectory_sha256={trajectory_payload['sha256']}",
            flush=True,
        )
    return trajectories, bank


def _run_bc_only_precheck(
    *,
    env: Any,
    policy: FrozenBCPolicy,
    provider: Any,
    recorder: OnlineExplicitFeedbackRecorder,
    protocol: Mapping[str, Any],
    scenario: Mapping[str, Any],
    seed: int,
    simulation_app: Any,
    proximal_monitor: ProximalArmProtocolMonitor,
    arm_joint_indices: Sequence[int],
    stop_requested: Callable[[], None],
) -> tuple[dict[str, Any], dict[str, Any], str, int]:
    cfg = dict(scenario["layout_precheck"])
    original_cbf = env._cbf_filter
    original_mode = env._physical_safety_mode
    env._cbf_filter = None
    env._physical_safety_mode = "none"
    try:
        for candidate_index in range(int(cfg["maximum_candidate_attempts"])):
            candidate_seed = int(seed + 10_000 + candidate_index)
            started_ns = time.monotonic_ns()
            obs, info = env.reset(seed=candidate_seed)
            obs, info, _tracking = _wait_for_tracking_and_clear(
                env=env,
                provider=provider,
                simulation_app=simulation_app,
                joint_count=len(env.robot.dof_names),
                timeout_s=60.0,
                stable_frames=int(cfg["hands_clear_confirmation_frames"]),
                clear_gap_m=float(cfg["hands_clear_gap_m"]),
                proximal_monitor=proximal_monitor,
                stop_requested=stop_requested,
            )
            scene, restoration, layout_id = _capture_scene_and_restoration(
                env, seed=candidate_seed
            )
            contaminated = False
            nominal_phase_references: dict[str, dict[str, Any]] = {}
            trigger_events = {
                str(phase): int(event)
                for phase, event in protocol["task"][
                    "crossing_trigger_event"
                ].items()
            }
            for _ in range(int(cfg["maximum_steps"])):
                stop_requested()
                snapshot = provider.observation_snapshot
                if snapshot is None:
                    raise CollectionAbort("BC precheck has no tracking snapshot")
                provider.watchdog.require_current(snapshot)
                event = int(info.get("controller_event", -1))
                for phase, trigger_event in trigger_events.items():
                    if (
                        event == trigger_event
                        and phase not in nominal_phase_references
                    ):
                        ee_position, _ee_orientation = _ee_pose(env)
                        joint_positions, _joint_velocities = measured_joint_state(
                            env.robot, joint_count=len(env.robot.dof_names)
                        )
                        nominal_phase_references[phase] = {
                            "phase": phase,
                            "controller_event": event,
                            "controller": "bc_only",
                            "ee_position_world_m": ee_position.tolist(),
                            "arm_joint_positions_rad": np.asarray(
                                joint_positions, dtype=float
                            )[np.asarray(arm_joint_indices, dtype=int)].tolist(),
                            "control_step": int(info.get("step", 0)),
                            "simulation_time_s": float(
                                info.get(
                                    "sim_time",
                                    getattr(env.world, "current_time", 0.0),
                                )
                            ),
                        }
                gaps = (
                    float(info.get("left_end_effector_surface_gap_m", -1.0)),
                    float(info.get("right_end_effector_surface_gap_m", -1.0)),
                )
                proximal_gaps = tuple(
                    proximal_monitor.closest_surface_gap_m(
                        getattr(snapshot, hand).position_world
                    )[0]
                    for hand in ("left", "right")
                )
                if (
                    not bool(info.get("geometry_valid", False))
                    or min(gaps) < float(cfg["hands_clear_gap_m"])
                    or min(proximal_gaps) < float(cfg["hands_clear_gap_m"])
                ):
                    contaminated = True
                    break
                action = policy.predict(np.asarray(obs, dtype=np.float32)).action
                obs, _reward, terminated, truncated, info = env.step(action)
                obs = np.asarray(obs, dtype=np.float32)
                if terminated or truncated:
                    terminal_snapshot = provider.observation_snapshot
                    if terminal_snapshot is None:
                        raise CollectionAbort(
                            "BC precheck terminal step has no tracking snapshot"
                        )
                    provider.watchdog.require_current(terminal_snapshot)
                    terminal_distal_gaps = (
                        float(
                            info.get(
                                "left_end_effector_surface_gap_m", -1.0
                            )
                        ),
                        float(
                            info.get(
                                "right_end_effector_surface_gap_m", -1.0
                            )
                        ),
                    )
                    terminal_proximal_gaps = tuple(
                        proximal_monitor.closest_surface_gap_m(
                            getattr(terminal_snapshot, hand).position_world
                        )[0]
                        for hand in ("left", "right")
                    )
                    if (
                        not bool(info.get("geometry_valid", False))
                        or min(terminal_distal_gaps)
                        < float(cfg["hands_clear_gap_m"])
                        or min(terminal_proximal_gaps)
                        < float(cfg["hands_clear_gap_m"])
                    ):
                        contaminated = True
                    break
            state = _strict_state(info)
            strict_success = bool(
                info.get("success", False) and state.get("success_latched", False)
            )
            release_observed = bool(
                strict_success and int(info.get("controller_event", -1)) >= 7
            )
            references_complete = set(nominal_phase_references) == set(
                trigger_events
            )
            restoration = dict(restoration)
            restoration["nominal_bc_phase_references"] = (
                nominal_phase_references
            )
            if contaminated:
                failure = "precheck_human_not_clear"
            elif not references_complete:
                failure = "precheck_nominal_phase_reference_incomplete:" + ",".join(
                    sorted(set(trigger_events) - set(nominal_phase_references))
                )
            elif not strict_success:
                terminal = str(info.get("task_terminal_reason", ""))
                failure = str(
                    state.get("failure_reason", "")
                    or ("" if terminal == "success" else terminal)
                    or "precheck_horizon"
                )
            else:
                failure = ""
            accepted = bool(
                strict_success
                and release_observed
                and references_complete
                and not contaminated
            )
            nominal_trajectories: dict[str, Any] = {}
            nominal_corridor_bank: dict[str, Any] = {}
            if accepted:
                try:
                    nominal_trajectories, nominal_corridor_bank = (
                        _calibrate_nominal_corridor_bank(
                            env=env,
                            policy=policy,
                            provider=provider,
                            protocol=protocol,
                            scenario=scenario,
                            restoration=restoration,
                            seed=candidate_seed,
                            simulation_app=simulation_app,
                            proximal_monitor=proximal_monitor,
                            arm_joint_indices=arm_joint_indices,
                            stop_requested=stop_requested,
                        )
                    )
                except (CollectionAbort, TrackingDropout):
                    raise
                except RuntimeError as error:
                    accepted = False
                    failure = (
                        "precheck_nominal_corridor_sweep_failed:"
                        f"{type(error).__name__}:{error}"
                    )
                else:
                    restoration["nominal_bc_phase_trajectories"] = (
                        nominal_trajectories
                    )
                    restoration["nominal_corridor_bank"] = (
                        nominal_corridor_bank
                    )
            corridor_bank_complete = bool(
                accepted
                and set(nominal_trajectories) == set(trigger_events)
                and set(nominal_corridor_bank) == set(trigger_events)
            )
            diagnostic_group = (
                "precheck_invalid_human_intrusion"
                if contaminated
                else (
                    cfg["accepted_group"]
                    if accepted
                    else (
                        "nominal_sweep_infeasible_diagnostic"
                        if strict_success and references_complete
                        else cfg["failed_group"]
                    )
                )
            )
            recorder.append_layout_precheck(
                {
                    "candidate_index": candidate_index,
                    "layout_id": layout_id,
                    "seed": candidate_seed,
                    "controller": "bc_only",
                    "policy_sha256": POLICY_SHA256,
                    "cbf_enabled": False,
                    "human_absent_or_clear": not contaminated,
                    "strict_success": strict_success,
                    "release_observed": release_observed,
                    "nominal_phase_references_complete": references_complete,
                    "nominal_phase_references": nominal_phase_references,
                    "nominal_corridor_bank_complete": corridor_bank_complete,
                    "nominal_phase_trajectories": nominal_trajectories,
                    "nominal_corridor_bank": nominal_corridor_bank,
                    "steps": int(info.get("step", 0)),
                    "diagnostic_group": diagnostic_group,
                    "failure_reason": failure,
                    "started_monotonic_ns": started_ns,
                    "ended_monotonic_ns": time.monotonic_ns(),
                    "source_restoration": _jsonable(restoration),
                }
            )
            print(
                f"[OnlineFeedback][BCPrecheck] candidate={candidate_index} "
                f"layout={layout_id} success={strict_success} clear={not contaminated} "
                f"steps={int(info.get('step', 0))} reason={failure or 'passed'}",
                flush=True,
            )
            env.world.pause()
            apply_controlled_hold(env.robot, joint_count=len(env.robot.dof_names))
            if accepted:
                return scene, restoration, layout_id, candidate_index
            if contaminated:
                provider.wait_until_ready(
                    simulation_app,
                    timeout_s=60.0,
                    stable_frames=6,
                    stop_requested=stop_requested,
                )
        raise CollectionAbort(
            "no BC-feasible single-cube layout passed frozen BC-only precheck"
        )
    finally:
        env._cbf_filter = original_cbf
        env._physical_safety_mode = original_mode
        if original_cbf is not None:
            original_cbf.reset()


def _experimental_phase(event: int, protocol: Mapping[str, Any]) -> str:
    for phase, events in protocol["task"]["phase_event_map"].items():
        if int(event) in [int(value) for value in events]:
            return str(phase)
    return "place_release" if int(event) >= 8 else "reach_approach"


def _object_state(env: Any) -> dict[str, np.ndarray]:
    cube_position, cube_orientation = env.active_cube.get_world_pose()
    goal_position, goal_orientation = env.place_target.get_world_pose()
    return {
        "cube_position_world_m": np.asarray(cube_position, dtype=np.float64)[:3],
        "cube_orientation_wxyz": np.asarray(cube_orientation, dtype=np.float64)[:4],
        "goal_position_world_m": np.asarray(goal_position, dtype=np.float64)[:3],
        "goal_orientation_wxyz": np.asarray(goal_orientation, dtype=np.float64)[:4],
    }


def _ee_pose(env: Any) -> tuple[np.ndarray, np.ndarray]:
    position, orientation = env.robot.end_effector.get_world_pose()
    return (
        np.asarray(position, dtype=np.float64)[:3],
        np.asarray(orientation, dtype=np.float64)[:4],
    )


def _dynamic_values(info: Mapping[str, Any]) -> tuple[float, float, bool]:
    dynamic = info.get("dynamic_safety", {})
    dynamic = dict(dynamic) if isinstance(dynamic, Mapping) else {}
    return (
        float(dynamic.get("min_ttc_s", 10.0)),
        max(0.0, float(dynamic.get("max_closing_speed_mps", 0.0))),
        bool(dynamic.get("dynamic_measurement_valid", False)),
    )


def _perform_logged_step(
    *,
    env: Any,
    policy: FrozenBCPolicy,
    provider: Any,
    watchdog: TrackingWatchdog,
    trace: ActionTraceRecorder,
    recorder: OnlineExplicitFeedbackRecorder,
    feedback: Any,
    button_source: Any,
    plan: TrialPlan,
    obs: np.ndarray,
    info: Mapping[str, Any],
    tracking: Any,
    arm_joint_indices: Sequence[int],
    control_mode: str,
    trial_state: str,
    advance_task_phase: bool,
    crossing_monitor: CrossingPathMonitor | None,
    detector: EncounterDetector | None,
    proximal_monitor: ProximalArmProtocolMonitor | None,
    protocol: Mapping[str, Any],
    stop_requested: Callable[[], None],
) -> LoggedStep:
    from v3_chan.rl.actions import zero_action

    stop_requested()
    preflight = provider.sample()
    preflight_ns = time.monotonic_ns()
    watchdog.require_current(preflight, now_monotonic_ns=preflight_ns)
    watchdog.require_current(tracking, now_monotonic_ns=preflight_ns)
    joint_count = len(env.robot.dof_names)
    positions_before, velocities_before = measured_joint_state(
        env.robot, joint_count=joint_count
    )
    ee_before, _ = _ee_pose(env)
    policy_output = policy.predict(np.asarray(obs, dtype=np.float32))
    controller_action = (
        policy_output.action if control_mode == "nominal_bc" else zero_action()
    )
    source_step = int(info.get("step", env.step_count))
    info_before = dict(info)
    trace.start_step(step_id=(plan.trial_index, source_step))
    try:
        obs_next, _reward, terminated, truncated, info_after = env.step(
            controller_action,
            advance_task_phase=bool(advance_task_phase),
        )
        action_trace = trace.finish_step()
    except Exception:
        trace.abort_step()
        apply_controlled_hold(env.robot, joint_count=joint_count)
        raise
    post_step_ns = time.monotonic_ns()
    wall_ns = time.time_ns()
    tracking_next = provider.observation_snapshot
    if tracking_next is None:
        raise CollectionAbort("control step produced no tracking provenance")
    watchdog.require_current(tracking_next, now_monotonic_ns=post_step_ns)
    positions_after, velocities_after = measured_joint_state(
        env.robot, joint_count=joint_count
    )
    validate_runtime_step(
        info_before=info_before,
        info_after=info_after,
        trace=action_trace,
        tracking=tracking,
        arm_joint_indices=arm_joint_indices,
    )
    ee_after, ee_orientation = _ee_pose(env)
    ee_velocity = (ee_after - ee_before) / max(float(env.physics_dt_s), 1e-9)
    simulation_time_s = float(
        info_after.get("sim_time", getattr(env.world, "current_time", 0.0))
    )
    cbf = info_after.get("physical_safety", {})
    cbf = dict(cbf) if isinstance(cbf, Mapping) else {}
    intervention_norm = float(cbf.get("intervention_norm_radps", 0.0))
    cbf_active = bool(cbf.get("active", False))
    minimum_gap = float(
        info_before.get("min_hand_end_effector_surface_gap", 10.0)
    )
    left_gap = float(info_before.get("left_end_effector_surface_gap_m", 10.0))
    right_gap = float(info_before.get("right_end_effector_surface_gap_m", 10.0))
    ttc_s, closing_speed, dynamic_measurement_valid = _dynamic_values(
        info_before
    )

    crossing_update = None
    proximal_by_hand: dict[str, tuple[float, str]] = {
        "left": (10.0, ""),
        "right": (10.0, ""),
    }
    crossing_payload = {
        "active": False,
        "state": "not_observed",
        "started_now": False,
        "intended_completed_now": False,
        "reverse_completed_now": False,
        "progress_fraction": 0.0,
        "perpendicular_deviation_m": 0.0,
        "intended_hand": plan.crossing_hand,
        "observed_hand": "",
        "runtime_feature_cutoff_available": detector is not None,
    }
    if crossing_monitor is not None:
        pose = getattr(tracking_next, plan.crossing_hand)
        valid_index = 1 if plan.crossing_hand == "left" else 2
        valid = bool(tracking_next.valid_mask[valid_index])
        crossing_update = crossing_monitor.observe(
            position_world=pose.position_world if valid else None,
            simulation_time_s=simulation_time_s,
            valid=valid,
        )
        crossing_payload.update(
            {
                "active": crossing_update.state
                in {"crossing", "completed", "extra_traversal"},
                "state": crossing_update.state,
                "started_now": crossing_update.crossing_started,
                "intended_completed_now": (
                    crossing_update.intended_crossing_completed
                ),
                "reverse_completed_now": (
                    crossing_update.reverse_crossing_completed
                ),
                "progress_fraction": 0.0
                if crossing_update.progress is None
                else crossing_update.progress,
                "perpendicular_deviation_m": 0.0
                if crossing_update.path_deviation_m is None
                else crossing_update.path_deviation_m,
                "observed_hand": plan.crossing_hand if valid else "",
            }
        )
    if proximal_monitor is not None:
        for hand, valid_index in (("left", 1), ("right", 2)):
            if bool(tracking_next.valid_mask[valid_index]):
                pose = getattr(tracking_next, hand)
                proximal_by_hand[hand] = proximal_monitor.closest_surface_gap_m(
                    pose.position_world
                )
    proximal_hand = min(
        proximal_by_hand, key=lambda hand: proximal_by_hand[hand][0]
    )
    proximal_gap, proximal_path = proximal_by_hand[proximal_hand]

    encounter_update = None
    if detector is not None:
        encounter_update = detector.observe(
            step=source_step,
            simulation_time_s=simulation_time_s,
            surface_gap_m=minimum_gap,
            ttc_s=ttc_s,
            closing=closing_speed > 0.0,
            cbf_delta_norm_radps=intervention_norm,
            cbf_active=cbf_active,
            dynamic_measurement_valid=dynamic_measurement_valid,
        )
    encounter_id = (
        "" if encounter_update is None else encounter_update.row_encounter_id
    )
    encounter_state = (
        "not_observed" if encounter_update is None else encounter_update.state
    )

    feedback_update = feedback.update(
        button_source.read(),
        _feedback_clocks(
            env, monotonic_ns=post_step_ns, unix_ns=wall_ns
        ),
    )
    for marker in feedback_update.realtime_markers:
        recorder.append_marker(
            RealtimeMarkerRecord(
                marker_id=marker.marker_id,
                trial_id=plan.trial_id,
                marker_type=marker.marker_type,
                simulation_time_s=marker.sim_time_s,
                monotonic_ns=marker.monotonic_ns,
                unix_ns=marker.unix_ns,
                control_step=marker.control_step,
                controller_hand=marker.controller_hand,
                button=marker.button,
            )
        )

    experimental_phase = _experimental_phase(
        int(info_before.get("controller_event", -1)), protocol
    )
    row = build_online_transition_row(
        source_step=source_step,
        observation_monotonic_ns=int(tracking.sample_monotonic_ns),
        preflight_monotonic_ns=preflight_ns,
        wall_time_unix_ns=wall_ns,
        obs_raw=np.asarray(obs, dtype=np.float32),
        obs_next=np.asarray(obs_next, dtype=np.float32),
        policy_output=policy_output,
        trace=action_trace,
        tracking=tracking,
        tracking_next=tracking_next,
        measured_positions_before=positions_before,
        measured_velocities_before=velocities_before,
        measured_positions_after=positions_after,
        measured_velocities_after=velocities_after,
        info_before=info_before,
        info_after=info_after,
        terminated=bool(terminated),
        truncated=bool(truncated),
        post_step_monotonic_ns=post_step_ns,
        arm_joint_indices=arm_joint_indices,
        encounter_id=encounter_id,
        encounter_state=encounter_state,
        controller_task_action=controller_action,
        controller_task_action_reason=(
            "frozen_bc_policy"
            if control_mode == "nominal_bc"
            else "current_ee_protocol_hold"
        ),
        control_mode=control_mode,
        trial_state=trial_state,
        trial_id=plan.trial_id,
        query_id=plan.query_id,
        experimental_phase=experimental_phase,
        phase_advance_enabled=bool(advance_task_phase),
        object_state=_object_state(env),
        ee_state={
            "position_world_m": ee_after,
            "orientation_wxyz": ee_orientation,
            "linear_velocity_world_m_s": ee_velocity,
        },
        crossing_state=crossing_payload,
        proximal_state={
            "left_gap_m": proximal_by_hand["left"][0],
            "right_gap_m": proximal_by_hand["right"][0],
            "left_collider_path": proximal_by_hand["left"][1],
            "right_collider_path": proximal_by_hand["right"][1],
        },
    )
    recorder.append_transition(row)
    return LoggedStep(
        obs=np.asarray(obs_next, dtype=np.float32),
        info=dict(info_after),
        tracking=tracking_next,
        terminated=bool(terminated),
        truncated=bool(truncated),
        source_step=source_step,
        result_step=int(info_after.get("step", source_step + 1)),
        simulation_time_s=simulation_time_s,
        monotonic_ns=post_step_ns,
        unix_ns=wall_ns,
        minimum_gap_m=minimum_gap,
        left_gap_m=left_gap,
        right_gap_m=right_gap,
        ttc_s=ttc_s,
        closing_speed_m_s=closing_speed,
        dynamic_measurement_valid=dynamic_measurement_valid,
        cbf_active=cbf_active,
        intervention_norm_rad_s=intervention_norm,
        crossing_update=crossing_update,
        encounter_update=encounter_update,
        questionnaire_result=feedback_update.questionnaire_result,
        realtime_markers=feedback_update.realtime_markers,
        proximal_gap_m=proximal_gap,
        proximal_collider_path=proximal_path,
        left_proximal_gap_m=proximal_by_hand["left"][0],
        right_proximal_gap_m=proximal_by_hand["right"][0],
        left_proximal_collider_path=proximal_by_hand["left"][1],
        right_proximal_collider_path=proximal_by_hand["right"][1],
    )


def _oriented_corridor(
    corridor: CrossingCorridor, direction: str
) -> CrossingCorridor:
    # Avatar/world convention: participant left is +Y and right is -Y.
    # Calibration's canonical tangent is +Y, hence its base start(-Y)->end(+Y)
    # is right-to-left.
    if direction == "right_to_left":
        return corridor
    if direction == "left_to_right":
        return replace(
            corridor,
            start_world_m=corridor.end_world_m.copy(),
            end_world_m=corridor.start_world_m.copy(),
            tangent_world=-corridor.tangent_world.copy(),
        )
    raise ValueError(f"unsupported crossing direction: {direction}")


def _corridor_from_mapping(value: Mapping[str, Any]) -> CrossingCorridor:
    semantics = str(value.get("geometry_query_semantics", ""))
    if semantics != "physx_protected_surface_to_hand_sphere":
        raise CollectionAbort("nominal corridor has unknown geometry semantics")
    corridor = CrossingCorridor(
        start_world_m=np.asarray(value.get("start_world_m", ()), dtype=float),
        end_world_m=np.asarray(value.get("end_world_m", ()), dtype=float),
        center_world_m=np.asarray(value.get("center_world_m", ()), dtype=float),
        tangent_world=np.asarray(value.get("tangent_world", ()), dtype=float),
        outward_normal_world=np.asarray(
            value.get("outward_normal_world", ()), dtype=float
        ),
        planned_minimum_surface_gap_m=float(
            value.get("planned_minimum_surface_gap_m", math.nan)
        ),
        closest_link=str(value.get("closest_link", "")),
        closest_collider_path=str(value.get("closest_collider_path", "")),
        calibration_iterations=int(value.get("calibration_iterations", 0)),
        geometry_query_semantics=semantics,
    )
    vectors = (
        corridor.start_world_m,
        corridor.end_world_m,
        corridor.center_world_m,
        corridor.tangent_world,
        corridor.outward_normal_world,
    )
    if any(vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in vectors):
        raise CollectionAbort("nominal corridor contains an invalid 3-vector")
    if (
        not math.isfinite(corridor.planned_minimum_surface_gap_m)
        or not corridor.closest_link
        or not corridor.closest_collider_path
        or corridor.calibration_iterations < 1
    ):
        raise CollectionAbort("nominal corridor provenance is incomplete")
    return corridor


def _nearest_nominal_trajectory_sample(
    trajectory: Mapping[str, Any], relative_time_s: float
) -> tuple[Mapping[str, Any], float]:
    samples = trajectory.get("samples", ())
    if not isinstance(samples, list) or not samples:
        raise CollectionAbort("nominal BC trajectory samples are unavailable")
    usable = [sample for sample in samples if isinstance(sample, Mapping)]
    if len(usable) != len(samples):
        raise CollectionAbort("nominal BC trajectory contains a malformed sample")
    expected = min(
        usable,
        key=lambda sample: abs(
            float(sample.get("relative_time_s", math.inf))
            - float(relative_time_s)
        ),
    )
    error = abs(
        float(expected.get("relative_time_s", math.inf))
        - float(relative_time_s)
    )
    if not math.isfinite(error):
        raise CollectionAbort("nominal BC trajectory time is invalid")
    return expected, error


def _nominal_state_errors(
    *,
    env: Any,
    expected: Mapping[str, Any],
    arm_joint_indices: Sequence[int],
) -> tuple[float, float]:
    ee_position, _orientation = _ee_pose(env)
    positions, _velocities = measured_joint_state(
        env.robot, joint_count=len(env.robot.dof_names)
    )
    expected_ee = np.asarray(
        expected.get("ee_position_world_m", ()), dtype=float
    ).reshape(-1)
    expected_arm = np.asarray(
        expected.get("arm_joint_positions_rad", ()), dtype=float
    ).reshape(-1)
    actual_arm = np.asarray(positions, dtype=float)[
        np.asarray(arm_joint_indices, dtype=int)
    ]
    if expected_ee.shape != (3,) or expected_arm.shape != actual_arm.shape:
        raise CollectionAbort("nominal BC trajectory state has invalid dimensions")
    if not np.all(np.isfinite(expected_ee)) or not np.all(np.isfinite(expected_arm)):
        raise CollectionAbort("nominal BC trajectory state is non-finite")
    return (
        float(np.linalg.norm(ee_position - expected_ee)),
        float(np.max(np.abs(actual_arm - expected_arm))),
    )


def _convert_encounter(
    record: Any, clocks_by_step: Mapping[int, tuple[float, int]]
) -> EncounterRecordV1:
    observed = sorted(
        (step, sim, mono) for step, (sim, mono) in clocks_by_step.items()
    )
    if not observed:
        raise RuntimeError("cannot serialize encounter without recorded clocks")

    def mono_for(step: int) -> int:
        exact = [item for item in observed if item[0] == step]
        return int(
            (exact[0] if exact else min(observed, key=lambda item: abs(item[0] - step)))[2]
        )

    window_steps = [
        step
        for step, sim, _ in observed
        if record.window_start_simulation_time_s
        <= sim
        <= record.window_end_simulation_time_s
    ]
    window_start = (
        min(window_steps) if window_steps else max(0, int(record.risk_onset_step))
    )
    window_end = (
        max(window_steps) + 1
        if window_steps
        else int(record.end_step_exclusive)
    )
    return EncounterRecordV1(
        encounter_id=record.encounter_id,
        trial_id=record.trial_id,
        onset_step=int(record.risk_onset_step),
        onset_confirmed_step=int(record.onset_confirmed_step),
        offset_step_exclusive=int(record.end_step_exclusive),
        onset_simulation_time_s=float(record.risk_onset_simulation_time_s),
        offset_simulation_time_s=float(record.offset_simulation_time_s),
        onset_monotonic_ns=mono_for(int(record.risk_onset_step)),
        offset_monotonic_ns=mono_for(int(record.offset_step)),
        window_start_step=int(window_start),
        window_end_step_exclusive=int(window_end),
        minimum_surface_gap_m=float(record.minimum_surface_gap_m),
        maximum_intervention_norm_rad_s=float(
            record.maximum_cbf_delta_norm_radps
        ),
        risk_onset_step=int(record.risk_onset_step),
        cbf_intervention_start_step=(
            -1
            if record.cbf_intervention_start_step is None
            else int(record.cbf_intervention_start_step)
        ),
        cbf_intervention_confirmed_step=(
            -1
            if record.cbf_intervention_confirmed_step is None
            else int(record.cbf_intervention_confirmed_step)
        ),
        encounter_timeout=bool(record.timeout),
        merged_reentry_count=int(record.reentry_merge_count),
    )


def _query_record(
    plan: TrialPlan, result: Any, encounter_id: str
) -> QueryRecord:
    completed = bool(result.study_complete)
    status = (
        "completed"
        if completed
        else ("timeout" if result.completion_status == "timed_out" else "no_response")
    )
    return QueryRecord(
        query_id=plan.query_id,
        trial_id=plan.trial_id,
        encounter_id=str(encounter_id or ""),
        issued_simulation_time_s=float(result.started_sim_time_s),
        issued_monotonic_ns=int(result.started_monotonic_ns),
        issued_unix_ns=int(result.started_unix_ns),
        completed_simulation_time_s=float(result.completed_sim_time_s),
        completed_monotonic_ns=int(result.completed_monotonic_ns),
        completed_unix_ns=int(result.completed_unix_ns),
        response_status=status,
        q1_response=str(result.q1_response or "") if completed else "",
        q2_collision_risk=int(result.q2 or 0) if completed else 0,
        q3_excessive_avoidance=int(result.q3 or 0) if completed else 0,
        q4_abrupt_unpredictable=int(result.q4 or 0) if completed else 0,
        q5_task_disruption=int(result.q5 or 0) if completed else 0,
        q6_confidence=int(result.q6 or 0) if completed else 0,
        rejection_reasons=(
            tuple(result.rejection_reason_ids or ()) if completed else ()
        ),
        response_latency_ms=(
            int(result.completed_monotonic_ns) - int(result.started_monotonic_ns)
        )
        / 1e6,
        input_source="vr_controller",
    )


def _set_instruction(feedback: Any, text: str) -> None:
    setter = getattr(feedback, "set_instruction", None)
    if callable(setter):
        setter(text)


def _run_trial(
    *,
    spec: TrialSpec,
    plan: TrialPlan,
    env: Any,
    policy: FrozenBCPolicy,
    provider: Any,
    watchdog: TrackingWatchdog,
    trace: ActionTraceRecorder,
    recorder: OnlineExplicitFeedbackRecorder,
    feedback: Any,
    button_source: Any,
    cue_visuals: CrossingCueVisuals,
    proximal_monitor: ProximalArmProtocolMonitor,
    protocol: Mapping[str, Any],
    scenario: Mapping[str, Any],
    restoration: Mapping[str, Any],
    initial_scene: Mapping[str, Any],
    layout_id: str,
    layout_precheck_index: int,
    arm_joint_indices: Sequence[int],
    simulation_app: Any,
    args: argparse.Namespace,
    stop_requested: Callable[[], None],
) -> None:
    trial_seed = int(args.seed + plan.trial_index)
    obs, info = env.reset(
        seed=trial_seed, source_restoration=dict(restoration)
    )
    # A world reset can restore the cue prim poses without resetting this
    # wrapper's Python state.  Synchronize both sides explicitly.
    cue_visuals.hide()
    obs, info, tracking = _wait_for_tracking_and_clear(
        env=env,
        provider=provider,
        simulation_app=simulation_app,
        joint_count=len(env.robot.dof_names),
        timeout_s=args.tracking_timeout_s,
        stable_frames=args.tracking_stable_frames,
        clear_gap_m=float(
            scenario["layout_precheck"]["hands_clear_gap_m"]
        ),
        proximal_monitor=proximal_monitor,
        stop_requested=stop_requested,
    )
    start_sim = float(
        info.get("sim_time", getattr(env.world, "current_time", 0.0))
    )
    start_mono, start_unix = time.monotonic_ns(), time.time_ns()
    recorder.start_trial(
        plan,
        seed=trial_seed,
        layout_id=layout_id,
        layout_precheck_index=layout_precheck_index,
        planned_gap_min_m=spec.planned_gap_min_m,
        planned_gap_max_m=spec.planned_gap_max_m,
        planned_target_gap_m=spec.planned_gap_target_m,
        start_simulation_time_s=start_sim,
        start_monotonic_ns=start_mono,
        start_unix_ns=start_unix,
        initial_scene=initial_scene,
    )
    dismiss = getattr(feedback, "dismiss_questionnaire", None)
    if callable(dismiss):
        dismiss()
    feedback.set_crossing_hand(None)
    trigger_event = int(
        protocol["task"]["crossing_trigger_event"][plan.task_phase]
    )
    crossing_cfg = dict(scenario["crossing"])
    staging_gap_m = float(scenario["layout_precheck"]["hands_clear_gap_m"])
    encounter_cfg = dict(protocol["encounter"])
    detector = EncounterDetector(
        EncounterDetectorConfig(
            activation_gap_m=float(encounter_cfg["activation_gap_m"]),
            onset_ttc_s=float(encounter_cfg["ttc_onset_s"]),
            intervention_norm_radps=float(
                encounter_cfg["intervention_onset_rad_s"]
            ),
            onset_confirmation_frames=int(
                encounter_cfg["onset_confirmation_frames"]
            ),
            intervention_confirmation_frames=3,
            clear_gap_m=float(encounter_cfg["clear_gap_m"]),
            clear_ttc_s=float(encounter_cfg["ttc_clear_s"]),
            clear_norm_radps=float(
                encounter_cfg["intervention_clear_rad_s"]
            ),
            clear_duration_s=float(encounter_cfg["clear_duration_s"]),
            reentry_merge_s=float(encounter_cfg["merge_reentry_s"]),
            timeout_s=float(encounter_cfg["timeout_s"]),
            pre_window_s=float(encounter_cfg["window_pre_s"]),
            post_window_s=float(encounter_cfg["window_post_s"]),
        ),
        trial_id=plan.trial_id,
    )
    crossing_monitor: CrossingPathMonitor | None = None
    corridor: CrossingCorridor | None = None
    nominal_reference_audit: dict[str, Any] | None = None
    nominal_sweep_audit: dict[str, Any] | None = None
    nominal_trajectory: Mapping[str, Any] | None = None
    start_recheck_audit: dict[str, Any] | None = None
    nominal_timeline_audit: dict[str, Any] = {
        "semantics": (
            "pre_cbf_actual_vs_bc_only_cue_latched_then_task_advancing_trajectory_v1"
        ),
        "compared_sample_count": 0,
        "maximum_time_alignment_error_s": 0.0,
        "maximum_ee_position_error_m": 0.0,
        "maximum_arm_joint_error_rad": 0.0,
        "comparison_closed_at_cbf_step": -1,
        "match": True,
    }
    core_encounter = None
    state = "wait_phase"
    cue_started_sim = math.inf
    crossing_completed_sim = math.inf
    crossing_started_sim = math.inf
    recovery_started_sim = math.inf
    query_started = False
    query_result = None
    task_active_elapsed_s = 0.0
    task_completion_active_elapsed_s: float | None = None
    clocks_by_step: dict[int, tuple[float, int]] = {}
    off_protocol_reasons: set[str] = set()
    minimum_gap = math.inf
    minimum_any_hand_gap = math.inf
    minimum_proximal_gap = math.inf
    closest_proximal_collider = ""
    cbf_any_active = False
    cbf_max_norm = 0.0
    intervention_duration = 0.0
    intervention_count = 0
    intervention_prior = False
    marker_counts = {
        "realtime_safety_concern": 0,
        "realtime_behavior_anomaly": 0,
    }
    _set_instruction(
        feedback,
        f"Trial {plan.trial_index + 1}: 로봇 작업을 관찰하세요. 손은 staging 위치에 유지하세요.",
    )

    while True:
        stop_requested()
        event = int(info.get("controller_event", -1))
        now_sim = float(
            info.get("sim_time", getattr(env.world, "current_time", 0.0))
        )
        if state == "wait_phase" and event == trigger_event:
            ee_position, _ = _ee_pose(env)
            references = restoration.get("nominal_bc_phase_references", {})
            trajectories = restoration.get(
                "nominal_bc_phase_trajectories", {}
            )
            corridor_bank = restoration.get("nominal_corridor_bank", {})
            reference = (
                references.get(plan.task_phase, {})
                if isinstance(references, Mapping)
                else {}
            )
            try:
                nominal_trajectory = trajectories[plan.task_phase]
                nominal_sweep_audit = dict(
                    corridor_bank[plan.task_phase][plan.direction][plan.speed][
                        plan.severity
                    ]
                )
                if not isinstance(nominal_trajectory, Mapping):
                    raise TypeError("trajectory is not a mapping")
                trajectory_hash = str(nominal_trajectory["sha256"])
                trajectory_body = {
                    key: value
                    for key, value in nominal_trajectory.items()
                    if key != "sha256"
                }
                if canonical_sha256(trajectory_body) != trajectory_hash:
                    raise ValueError("trajectory SHA-256 mismatch")
                sweep_hash = str(nominal_sweep_audit["sweep_audit_sha256"])
                sweep_body = {
                    key: value
                    for key, value in nominal_sweep_audit.items()
                    if key != "sweep_audit_sha256"
                }
                if canonical_sha256(
                    {"nominal_bc_sweep": sweep_body}
                ) != sweep_hash:
                    raise ValueError("sweep audit SHA-256 mismatch")
                if str(nominal_sweep_audit["trajectory_sha256"]) != trajectory_hash:
                    raise ValueError("sweep/trajectory relation mismatch")
                corridor = _corridor_from_mapping(
                    nominal_sweep_audit["corridor"]
                )
                reference_ee = np.asarray(
                    reference["ee_position_world_m"], dtype=float
                ).reshape(3)
                reference_joints = np.asarray(
                    reference["arm_joint_positions_rad"], dtype=float
                ).reshape(len(arm_joint_indices))
                current_positions, _current_velocities = measured_joint_state(
                    env.robot, joint_count=len(env.robot.dof_names)
                )
                current_arm = np.asarray(current_positions, dtype=float)[
                    np.asarray(arm_joint_indices, dtype=int)
                ]
                ee_error = float(np.linalg.norm(ee_position - reference_ee))
                joint_error = float(
                    np.max(np.abs(current_arm - reference_joints))
                )
            except (KeyError, TypeError, ValueError) as error:
                raise CollectionAbort(
                    "accepted layout lacks a valid time-aligned BC-only "
                    "nominal corridor/reference"
                ) from error
            reference_cfg = scenario["nominal_bc_reference"]
            expected_range = [
                float(spec.planned_gap_min_m),
                float(spec.planned_gap_max_m),
            ]
            sweep_exact = bool(
                nominal_sweep_audit.get("semantics")
                == NOMINAL_BC_SWEEP_SEMANTICS
                and nominal_sweep_audit.get("phase") == plan.task_phase
                and nominal_sweep_audit.get("severity") == plan.severity
                and nominal_sweep_audit.get("hand") == plan.crossing_hand
                and nominal_sweep_audit.get("crossing_direction")
                == plan.direction
                and math.isclose(
                    float(nominal_sweep_audit.get("speed_m_s", math.nan)),
                    float(crossing_cfg["speed_target_m_s"][plan.speed]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and nominal_sweep_audit.get("allowed_gap_range_m")
                == expected_range
                and math.isclose(
                    float(
                        nominal_sweep_audit.get("target_gap_m", math.nan)
                    ),
                    float(spec.planned_gap_target_m),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and bool(nominal_sweep_audit.get("geometry_valid", False))
                and float(spec.planned_gap_min_m)
                <= corridor.planned_minimum_surface_gap_m
                <= float(spec.planned_gap_max_m)
            )
            if not sweep_exact:
                raise CollectionAbort(
                    "nominal BC sweep does not match the scheduled condition"
                )
            reference_match = bool(
                str(reference.get("phase", "")) == plan.task_phase
                and str(reference.get("controller", "")) == "bc_only"
                and int(reference.get("controller_event", -1)) == trigger_event
                and ee_error
                <= float(reference_cfg["maximum_ee_position_error_m"])
                and joint_error
                <= float(reference_cfg["maximum_arm_joint_error_rad"])
            )
            nominal_reference_audit = {
                "semantics": str(reference_cfg["trigger_state_semantics"]),
                "phase": plan.task_phase,
                "controller": str(reference.get("controller", "")),
                "controller_event": int(
                    reference.get("controller_event", -1)
                ),
                "precheck_control_step": int(reference.get("control_step", -1)),
                "precheck_simulation_time_s": float(
                    reference.get("simulation_time_s", -1.0)
                ),
                "ee_position_error_m": ee_error,
                "maximum_arm_joint_error_rad": joint_error,
                "match": reference_match,
            }
            if not reference_match:
                off_protocol_reasons.add("nominal_bc_phase_reference_mismatch")
            crossing_monitor = CrossingPathMonitor(
                start_position_world=corridor.start_world_m,
                end_position_world=corridor.end_world_m,
                path_deviation_threshold_m=float(
                    crossing_cfg["maximum_path_deviation_m"]
                ),
                expected_speed_mps=float(
                    crossing_cfg["speed_target_m_s"][plan.speed]
                ),
                relative_speed_tolerance=float(
                    crossing_cfg["speed_tolerance_fraction"]
                ),
                start_radius_m=float(crossing_cfg["start_radius_m"]),
                end_radius_m=float(crossing_cfg["end_radius_m"]),
                minimum_progress_fraction=float(
                    crossing_cfg["minimum_progress_fraction"]
                ),
                maximum_reverse_progress_m=float(
                    crossing_cfg["maximum_reverse_progress_m"]
                ),
            )
            cue_started_sim = now_sim
            feedback.set_crossing_hand(plan.crossing_hand)
            cue_visuals.show(
                corridor,
                simulation_time_s=now_sim + float(crossing_cfg["cue_lead_s"]),
                speed_m_s=float(crossing_cfg["speed_target_m_s"][plan.speed]),
            )
            _set_instruction(
                feedback,
                f"{plan.task_phase} | {plan.crossing_hand}손 | {plan.direction} | {plan.speed}\n"
                "초록 시작점으로 이동해 START까지 대기하세요. "
                "START 후 노란 guide 속도로 빨강 끝점까지 한 번만 crossing 하세요.",
            )
            state = "pre_crossing"

        # The controlled crossing is an intervention into the *running* frozen
        # task policy.  Latch only during the pre-crossing cue so the assigned
        # phase cannot disappear before START.  From START through response
        # and recovery, advance the normal task controller under frozen BC;
        # only the post-recovery questionnaire pauses progression with a
        # current-EE hold while physics and CBF continue.
        control_mode = "protocol_hold" if state == "query" else "nominal_bc"
        advance = state in {
            "wait_phase",
            "crossing",
            "response",
            "recovery",
            "resume_task",
        }
        monitor_for_step = (
            crossing_monitor
            if state in {"pre_crossing", "crossing", "response", "recovery"}
            else None
        )
        # The cue-lead rows are retained as pre-window provenance, but risk
        # onset is armed only at START.  Otherwise moving into the start gate
        # could become the encounter instead of the controlled traversal.
        detector_for_step = (
            detector
            if state in {"crossing", "response"} and core_encounter is None
            else None
        )
        state_during_step = state
        step = _perform_logged_step(
            env=env,
            policy=policy,
            provider=provider,
            watchdog=watchdog,
            trace=trace,
            recorder=recorder,
            feedback=feedback,
            button_source=button_source,
            plan=plan,
            obs=obs,
            info=info,
            tracking=tracking,
            arm_joint_indices=arm_joint_indices,
            control_mode=control_mode,
            trial_state=("query_hold" if state == "query" else state),
            advance_task_phase=advance,
            crossing_monitor=monitor_for_step,
            detector=detector_for_step,
            proximal_monitor=proximal_monitor,
            protocol=protocol,
            stop_requested=stop_requested,
        )
        obs, info, tracking = step.obs, step.info, step.tracking
        if state_during_step != "query":
            task_active_elapsed_s += float(env.physics_dt_s)
        if (
            bool(info.get("success", False))
            and task_completion_active_elapsed_s is None
        ):
            task_completion_active_elapsed_s = task_active_elapsed_s
        clocks_by_step[step.source_step] = (
            step.simulation_time_s,
            step.monotonic_ns,
        )
        cue_visuals.update(step.simulation_time_s)
        if (
            nominal_trajectory is not None
            and state_during_step in {"pre_crossing", "crossing"}
        ):
            if step.cbf_active or step.intervention_norm_rad_s > 0.01:
                if nominal_timeline_audit["comparison_closed_at_cbf_step"] < 0:
                    nominal_timeline_audit[
                        "comparison_closed_at_cbf_step"
                    ] = step.source_step
            elif nominal_timeline_audit["comparison_closed_at_cbf_step"] < 0:
                relative_time_s = step.simulation_time_s - cue_started_sim
                samples = nominal_trajectory.get("samples", ())
                final_relative_time_s = (
                    float(samples[-1].get("relative_time_s", -math.inf))
                    if isinstance(samples, list)
                    and samples
                    and isinstance(samples[-1], Mapping)
                    else -math.inf
                )
                if relative_time_s <= final_relative_time_s + 1e-12:
                    expected_state, time_error = (
                        _nearest_nominal_trajectory_sample(
                            nominal_trajectory, relative_time_s
                        )
                    )
                    ee_state_error, joint_state_error = _nominal_state_errors(
                        env=env,
                        expected=expected_state,
                        arm_joint_indices=arm_joint_indices,
                    )
                    nominal_timeline_audit["compared_sample_count"] += 1
                    nominal_timeline_audit[
                        "maximum_time_alignment_error_s"
                    ] = max(
                        nominal_timeline_audit[
                            "maximum_time_alignment_error_s"
                        ],
                        time_error,
                    )
                    nominal_timeline_audit[
                        "maximum_ee_position_error_m"
                    ] = max(
                        nominal_timeline_audit[
                            "maximum_ee_position_error_m"
                        ],
                        ee_state_error,
                    )
                    nominal_timeline_audit[
                        "maximum_arm_joint_error_rad"
                    ] = max(
                        nominal_timeline_audit[
                            "maximum_arm_joint_error_rad"
                        ],
                        joint_state_error,
                    )
                    reference_cfg = scenario["nominal_bc_reference"]
                    row_match = bool(
                        time_error
                        <= float(
                            reference_cfg[
                                "maximum_sweep_sample_interval_s"
                            ]
                        )
                        and ee_state_error
                        <= float(reference_cfg["maximum_ee_position_error_m"])
                        and joint_state_error
                        <= float(reference_cfg["maximum_arm_joint_error_rad"])
                    )
                    nominal_timeline_audit["match"] = bool(
                        nominal_timeline_audit["match"] and row_match
                    )
                    if not row_match:
                        off_protocol_reasons.add(
                            "nominal_timeline_drift_before_cbf"
                        )
        for marker in step.realtime_markers:
            marker_counts[marker.marker_type] += 1
        if state == "wait_phase" and (
            step.minimum_gap_m <= staging_gap_m
            or step.proximal_gap_m <= staging_gap_m
            or step.cbf_active
            or step.intervention_norm_rad_s > 0.01
        ):
            off_protocol_reasons.add("pre_cue_human_intrusion")
        if state in {"pre_crossing", "crossing", "response", "recovery"}:
            intended_gap = (
                step.left_gap_m
                if plan.crossing_hand == "left"
                else step.right_gap_m
            )
            minimum_gap = min(minimum_gap, intended_gap)
            minimum_any_hand_gap = min(
                minimum_any_hand_gap, step.minimum_gap_m
            )
            if step.proximal_gap_m < minimum_proximal_gap:
                minimum_proximal_gap = step.proximal_gap_m
                closest_proximal_collider = step.proximal_collider_path
            cbf_any_active = cbf_any_active or step.cbf_active
            cbf_max_norm = max(cbf_max_norm, step.intervention_norm_rad_s)
            intervening = step.intervention_norm_rad_s > 0.01
            intervention_duration += (
                float(env.physics_dt_s) if intervening else 0.0
            )
            if intervening and not intervention_prior:
                intervention_count += 1
            intervention_prior = intervening
            opposite_gap = (
                step.right_gap_m
                if plan.crossing_hand == "left"
                else step.left_gap_m
            )
            if opposite_gap <= float(protocol["cbf"]["activation_gap_m"]):
                off_protocol_reasons.add("non_crossing_hand_intrusion")
            opposite_proximal_gap = (
                step.right_proximal_gap_m
                if plan.crossing_hand == "left"
                else step.left_proximal_gap_m
            )
            if opposite_proximal_gap <= float(
                protocol["cbf"]["activation_gap_m"]
            ):
                off_protocol_reasons.add(
                    "non_crossing_hand_proximal_intrusion"
                )
            if step.proximal_gap_m <= 0.13:
                off_protocol_reasons.add("proximal_unprotected_crossing")
                if step.proximal_collider_path:
                    off_protocol_reasons.add(
                        "proximal_collider:" + step.proximal_collider_path
                    )
        if (
            step.encounter_update is not None
            and step.encounter_update.closed is not None
        ):
            core_encounter = step.encounter_update.closed
        risk_now = bool(
            step.minimum_gap_m <= float(encounter_cfg["activation_gap_m"])
            or (
                step.dynamic_measurement_valid
                and
                step.closing_speed_m_s > 0.0
                and 0.0 <= step.ttc_s <= float(encounter_cfg["ttc_onset_s"])
            )
            or step.intervention_norm_rad_s
            >= float(encounter_cfg["intervention_onset_rad_s"])
        )
        if (
            core_encounter is not None
            and risk_now
            and state in {"crossing", "response", "recovery"}
            and (
                step.encounter_update is None
                or step.encounter_update.row_encounter_id
                != core_encounter.encounter_id
            )
        ):
            # A >1 s post-offset risk episode would be a second encounter in a
            # one-crossing trial.  Preserve its raw per-step evidence, exclude
            # the trial, and keep the primary query relation unambiguous.
            off_protocol_reasons.add("multiple_encounter_risk_after_close")
        elif state == "recovery" and core_encounter is None and risk_now:
            off_protocol_reasons.add("risk_started_during_recovery")

        if state == "wait_phase":
            if step.terminated or step.truncated:
                off_protocol_reasons.add("task_ended_before_crossing_phase")
                state = "recovery"
                recovery_started_sim = step.simulation_time_s
        elif state == "pre_crossing":
            early_risk = bool(
                step.minimum_gap_m <= float(encounter_cfg["activation_gap_m"])
                or (
                    step.dynamic_measurement_valid
                    and
                    step.closing_speed_m_s > 0.0
                    and 0.0 <= step.ttc_s <= float(encounter_cfg["ttc_onset_s"])
                )
                or step.intervention_norm_rad_s
                >= float(encounter_cfg["intervention_onset_rad_s"])
            )
            if early_risk or step.cbf_active:
                off_protocol_reasons.add("early_encounter_before_crossing")
            if (
                step.crossing_update is not None
                and step.crossing_update.intended_crossing_completed
            ):
                off_protocol_reasons.add("crossing_completed_before_start_cue")
            if (
                step.simulation_time_s - cue_started_sim
                >= float(crossing_cfg["cue_lead_s"])
            ):
                assert crossing_monitor is not None
                assert corridor is not None
                assert nominal_sweep_audit is not None
                assert nominal_trajectory is not None
                reference_cfg = scenario["nominal_bc_reference"]
                coverage = nominal_sweep_audit.get("coverage", {})
                if not isinstance(coverage, Mapping):
                    raise CollectionAbort(
                        "nominal sweep START coverage audit is malformed"
                    )
                expected_start_relative_s = float(
                    nominal_sweep_audit.get(
                        "crossing_start_query_simulation_time_s", math.nan
                    )
                ) - float(
                    coverage.get("phase_start_simulation_time_s", math.nan)
                )
                actual_start_relative_s = (
                    step.simulation_time_s - cue_started_sim
                )
                expected_start_state, state_time_error = (
                    _nearest_nominal_trajectory_sample(
                        nominal_trajectory, actual_start_relative_s
                    )
                )
                start_ee_error, start_joint_error = _nominal_state_errors(
                    env=env,
                    expected=expected_start_state,
                    arm_joint_indices=arm_joint_indices,
                )
                start_result = env.safety_geometry.evaluate_hand(
                    plan.crossing_hand, corridor.start_world_m
                )
                actual_start_gap = float(
                    getattr(start_result, "surface_gap_m", math.nan)
                )
                expected_start_gap = float(
                    nominal_sweep_audit.get(
                        "crossing_start_surface_gap_m", math.nan
                    )
                )
                start_gap_error = abs(actual_start_gap - expected_start_gap)
                scheduled_time_error = abs(
                    actual_start_relative_s - expected_start_relative_s
                )
                start_geometry_valid = bool(
                    getattr(start_result, "geometry_valid", False)
                    and math.isfinite(actual_start_gap)
                    and str(getattr(start_result, "closest_link", ""))
                    and str(
                        getattr(start_result, "closest_collider_path", "")
                    )
                )
                start_match = bool(
                    start_geometry_valid
                    and math.isfinite(expected_start_relative_s)
                    and scheduled_time_error
                    <= float(reference_cfg["maximum_sweep_sample_interval_s"])
                    and state_time_error
                    <= float(reference_cfg["maximum_sweep_sample_interval_s"])
                    and start_ee_error
                    <= float(reference_cfg["maximum_ee_position_error_m"])
                    and start_joint_error
                    <= float(reference_cfg["maximum_arm_joint_error_rad"])
                    and start_gap_error
                    <= float(reference_cfg["maximum_start_gap_error_m"])
                )
                start_recheck_audit = {
                    "semantics": "actual_start_vs_time_aligned_nominal_sweep_v1",
                    "actual_relative_time_s": actual_start_relative_s,
                    "expected_query_relative_time_s": expected_start_relative_s,
                    "scheduled_time_error_s": scheduled_time_error,
                    "nearest_state_time_error_s": state_time_error,
                    "actual_start_surface_gap_m": actual_start_gap,
                    "expected_start_surface_gap_m": expected_start_gap,
                    "start_surface_gap_error_m": start_gap_error,
                    "ee_position_error_m": start_ee_error,
                    "maximum_arm_joint_error_rad": start_joint_error,
                    "actual_closest_link": str(
                        getattr(start_result, "closest_link", "")
                    ),
                    "actual_closest_collider_path": str(
                        getattr(start_result, "closest_collider_path", "")
                    ),
                    "geometry_valid": start_geometry_valid,
                    "match": start_match,
                }
                if not start_match:
                    off_protocol_reasons.add(
                        "nominal_start_recheck_failed"
                    )
                hand_pose = getattr(step.tracking, plan.crossing_hand)
                # Pre-START samples are useful audit rows, but dwell/jitter at
                # the green marker must not bias traversal speed, reverse
                # progress, or the one-crossing count.  Re-observe the current
                # pose after reset so START readiness uses the exact same
                # radial+axial gate predicate as the trial measurement.
                crossing_monitor.reset()
                start_update = crossing_monitor.observe(
                    position_world=hand_pose.position_world,
                    simulation_time_s=step.simulation_time_s,
                    valid=True,
                )
                if not start_update.crossing_started:
                    off_protocol_reasons.add(
                        "hand_not_at_start_gate_on_start_cue"
                    )
                state = "crossing"
                crossing_started_sim = step.simulation_time_s
                _set_instruction(
                    feedback,
                    "START — guide를 따라 지정 손을 한 번 crossing 하세요.",
                )
        elif state == "crossing":
            if (
                step.crossing_update is not None
                and step.crossing_update.intended_crossing_completed
            ):
                crossing_completed_sim = step.simulation_time_s
                state = "response"
                _set_instruction(
                    feedback,
                    "손은 끝점 밖으로 이동하고 로봇 반응을 계속 관찰하세요.",
                )
            elif (
                step.simulation_time_s - crossing_started_sim
                > float(crossing_cfg["crossing_timeout_s"])
            ):
                off_protocol_reasons.add("crossing_timeout")
                crossing_completed_sim = step.simulation_time_s
                state = "response"
        elif state == "response":
            encounter_timed_out = bool(
                step.encounter_update is not None
                and step.encounter_update.timed_out_now
            )
            no_encounter_settle = (
                not detector.active
                and step.simulation_time_s - crossing_completed_sim >= 0.5
            )
            if encounter_timed_out:
                core_encounter = detector.flush(
                    step=step.source_step,
                    simulation_time_s=step.simulation_time_s,
                    terminal_status="encounter_timeout",
                )
                recovery_started_sim = step.simulation_time_s
                state = "recovery"
                _set_instruction(
                    feedback,
                    "Recovery 관찰 중 — 로봇 반응을 계속 보세요.",
                )
            elif core_encounter is not None or no_encounter_settle:
                recovery_started_sim = step.simulation_time_s
                state = "recovery"
                _set_instruction(
                    feedback,
                    "Recovery 관찰 중 — 로봇 반응을 계속 보세요.",
                )
            elif (
                step.simulation_time_s - crossing_completed_sim
                > float(encounter_cfg["timeout_s"])
            ):
                off_protocol_reasons.add("response_settle_timeout")
                core_encounter = detector.flush(
                    step=step.source_step,
                    simulation_time_s=step.simulation_time_s,
                    terminal_status="encounter_timeout",
                )
                recovery_started_sim = step.simulation_time_s
                state = "recovery"
        elif state == "recovery":
            if (
                step.simulation_time_s - recovery_started_sim
                >= float(crossing_cfg["recovery_observation_s"])
            ):
                cue_visuals.hide()
                feedback.start_questionnaire(
                    _feedback_clocks(
                        env,
                        monotonic_ns=time.monotonic_ns(),
                        unix_ns=time.time_ns(),
                    ),
                    context_id=plan.trial_id,
                    timeout_s=float(protocol["feedback"]["query_timeout_s"]),
                )
                query_started = True
                state = "query"
        elif state == "query":
            if step.questionnaire_result is not None:
                query_result = step.questionnaire_result
                break

        if (
            state not in {"query", "resume_task"}
            and (step.terminated or step.truncated)
            and not bool(info.get("success", False))
            and not query_started
            and state != "recovery"
        ):
            off_protocol_reasons.add("task_terminal_before_query")
        if (
            int(info.get("step", 0)) >= int(env.config.max_episode_steps) - 2
            and state != "query"
        ):
            raise CollectionAbort(
                "trial exhausted control horizon before mandatory query"
            )

    assert query_result is not None
    query = _query_record(
        plan,
        query_result,
        "" if core_encounter is None else core_encounter.encounter_id,
    )
    for answer in query_result.answers:
        answer_payload = answer.as_dict()
        answer_payload["question_id"] = {
            "q1_modification_needed": Q1_ID,
            "q2": "q2_collision_risk",
            "q3": "q3_excessive_avoidance",
            "q4": "q4_abrupt_unpredictable",
            "q5": "q5_task_disruption",
            "q6": "q6_confidence",
            "rejection_reasons": "rejection_reasons",
        }.get(str(answer.question_id), str(answer.question_id))
        recorder.append_query_answer(
            query_id=plan.query_id,
            trial_id=plan.trial_id,
            answer=answer_payload,
        )
    recorder.append_query(query)
    feedback.set_crossing_hand(None)
    _set_instruction(feedback, "응답이 저장되었습니다. 다음 trial을 준비합니다.")

    if (
        not args.reset_after_query
        and bool(protocol["task"]["complete_task_after_query"])
    ):
        while not (
            bool(info.get("success", False))
            or str(info.get("task_terminal_reason", ""))
            or int(info.get("step", 0)) >= int(env.config.max_episode_steps)
        ):
            step = _perform_logged_step(
                env=env,
                policy=policy,
                provider=provider,
                watchdog=watchdog,
                trace=trace,
                recorder=recorder,
                feedback=feedback,
                button_source=button_source,
                plan=plan,
                obs=obs,
                info=info,
                tracking=tracking,
                arm_joint_indices=arm_joint_indices,
                control_mode="nominal_bc",
                trial_state="resume_task",
                advance_task_phase=True,
                crossing_monitor=None,
                detector=None,
                proximal_monitor=None,
                protocol=protocol,
                stop_requested=stop_requested,
            )
            obs, info, tracking = step.obs, step.info, step.tracking
            task_active_elapsed_s += float(env.physics_dt_s)
            if (
                bool(info.get("success", False))
                and task_completion_active_elapsed_s is None
            ):
                task_completion_active_elapsed_s = task_active_elapsed_s
            clocks_by_step[step.source_step] = (
                step.simulation_time_s,
                step.monotonic_ns,
            )
            if step.terminated or step.truncated:
                break

    if core_encounter is None:
        core_encounter = detector.flush(
            step=max(0, int(info.get("step", 1)) - 1),
            simulation_time_s=float(
                info.get("sim_time", getattr(env.world, "current_time", 0.0))
            ),
            terminal_status="trial_end",
        )
    encounter_v1 = None
    if core_encounter is not None:
        encounter_v1 = _convert_encounter(core_encounter, clocks_by_step)
        recorder.append_encounter(encounter_v1)
    encounter_id = "" if encounter_v1 is None else encounter_v1.encounter_id
    if encounter_id != query.encounter_id:
        raise CollectionAbort(
            "query/encounter relation changed after query completion"
        )

    metrics = (
        crossing_monitor.finalize(
            additional_off_protocol_reasons=off_protocol_reasons
        )
        if crossing_monitor is not None
        else None
    )
    if metrics is None:
        crossing_count = 0
        rms_dev = max_dev = actual_speed = 0.0
        off_protocol_reasons.add("crossing_not_initialized")
    else:
        crossing_count = metrics.intended_crossing_count
        rms_dev = _finite_or_zero(metrics.rms_path_deviation_m)
        max_dev = _finite_or_zero(metrics.maximum_path_deviation_m)
        actual_speed = _finite_or_zero(metrics.mean_speed_mps)
        off_protocol_reasons.update(metrics.off_protocol_reasons)
    if core_encounter is not None and core_encounter.timeout:
        off_protocol_reasons.add("encounter_timeout")
    if plan.severity == "control" and (cbf_any_active or core_encounter is not None):
        off_protocol_reasons.add("control_condition_cbf_or_risk_activation")
    if query.response_status != "completed":
        off_protocol_reasons.add("mandatory_query_incomplete")
    if nominal_sweep_audit is None or nominal_reference_audit is None:
        off_protocol_reasons.add("nominal_calibration_audit_unavailable")
    if start_recheck_audit is None or not bool(
        start_recheck_audit.get("match", False)
    ):
        off_protocol_reasons.add("nominal_start_recheck_failed")
    if (
        int(nominal_timeline_audit["compared_sample_count"]) < 1
        or not bool(nominal_timeline_audit["match"])
    ):
        off_protocol_reasons.add("nominal_timeline_audit_failed")
    protocol_valid = bool(
        crossing_count == 1
        and not off_protocol_reasons
        and query.response_status == "completed"
    )
    strict = _strict_state(info)
    task_success = bool(info.get("success", False))
    terminal_reason = _normalized_task_failure_reason(
        info,
        strict_state=strict,
        task_success=task_success,
        reset_after_query=bool(args.reset_after_query),
        max_episode_steps=int(env.config.max_episode_steps),
    )
    end_sim = float(
        info.get("sim_time", getattr(env.world, "current_time", 0.0))
    )
    recorder.end_trial(
        {
            "protocol_valid": protocol_valid,
            "actual_crossing_count": crossing_count,
            "actual_path_deviation_rms_m": rms_dev,
            "actual_path_deviation_max_m": max_dev,
            "actual_crossing_speed_m_s": actual_speed,
            "actual_minimum_gap_m": (
                10.0 if not math.isfinite(minimum_gap) else minimum_gap
            ),
            "actual_minimum_any_hand_gap_m": (
                10.0
                if not math.isfinite(minimum_any_hand_gap)
                else minimum_any_hand_gap
            ),
            "actual_minimum_proximal_gap_m": (
                10.0
                if not math.isfinite(minimum_proximal_gap)
                else minimum_proximal_gap
            ),
            "closest_proximal_collider": closest_proximal_collider,
            "off_protocol": bool(off_protocol_reasons),
            "off_protocol_reasons": sorted(off_protocol_reasons),
            "cbf_activated": cbf_any_active,
            "cbf_total_intervention_duration_s": intervention_duration,
            "cbf_max_intervention_norm_rad_s": cbf_max_norm,
            "cbf_intervention_count": intervention_count,
            "task_success": task_success,
            "task_failure_reason": terminal_reason,
            "object_drop": (
                "drop" in terminal_reason.lower()
                or "grasp_lost" in terminal_reason.lower()
            ),
            "grasp_outcome": (
                "grasped" if bool(strict.get("grasp_observed", False)) else "not_observed"
            ),
            "release_outcome": (
                "released_and_settled"
                if bool(strict.get("success_latched", False))
                else ("reset_after_query" if args.reset_after_query else "not_settled")
            ),
            "task_completion_observed": (
                task_completion_active_elapsed_s is not None
            ),
            "completion_time_s": (
                -1.0
                if task_completion_active_elapsed_s is None
                else max(0.0, task_completion_active_elapsed_s)
            ),
            "completion_time_semantics": (
                "active_task_simulation_time_excluding_query_hold_v1"
            ),
            "trial_simulation_duration_s": max(0.0, end_sim - start_sim),
            "trial_wall_duration_s": max(
                0.0, (time.monotonic_ns() - start_mono) / 1e9
            ),
            "recovery_time_s": (
                0.0
                if not math.isfinite(recovery_started_sim)
                else max(
                    0.0,
                    query.issued_simulation_time_s - recovery_started_sim,
                )
            ),
            "risk_onset_step": (
                -1 if encounter_v1 is None else encounter_v1.risk_onset_step
            ),
            "cbf_intervention_start_step": (
                -1
                if encounter_v1 is None
                else encounter_v1.cbf_intervention_start_step
            ),
            "cbf_intervention_confirmed_step": (
                -1
                if encounter_v1 is None
                else encounter_v1.cbf_intervention_confirmed_step
            ),
            "realtime_safety_marker_count": marker_counts[
                "realtime_safety_concern"
            ],
            "realtime_behavior_anomaly_marker_count": marker_counts[
                "realtime_behavior_anomaly"
            ],
            "planned_corridor_json": (
                "{}"
                if corridor is None
                else json.dumps(
                    {
                        **corridor.as_dict(),
                        "nominal_bc_reference": nominal_reference_audit,
                        "nominal_bc_sweep": nominal_sweep_audit,
                        "start_recheck": start_recheck_audit,
                        "pre_cbf_timeline_audit": nominal_timeline_audit,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "end_simulation_time_s": end_sim,
            "end_monotonic_ns": time.monotonic_ns(),
            "end_unix_ns": time.time_ns(),
        }
    )
    cue_visuals.hide()
    print(
        f"[OnlineFeedback] trial={plan.trial_index:03d} "
        f"phase={plan.task_phase} severity={plan.severity} "
        f"crossings={crossing_count} encounter={encounter_id or 'null'} "
        f"q1={query.q1_response or 'missing'} "
        f"off_protocol={bool(off_protocol_reasons)} task_success={task_success}",
        flush=True,
    )


def _run_collection(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    scenario: dict[str, Any],
    specs: Sequence[TrialSpec],
    plans: Sequence[TrialPlan],
) -> Path:
    global simulation_app
    output = (args.output or _default_output(args)).expanduser().resolve()
    runtime_report = validate_runtime_contract(
        PROJECT_ROOT, check_checkpoint_metadata=True
    )
    if not runtime_report["valid"]:
        raise RuntimeError(
            "frozen runtime contract failed: "
            + "; ".join(runtime_report["failures"])
        )
    if runtime_report.get("contract_sha256") != RUNTIME_CONTRACT_SHA256:
        raise RuntimeError("runtime contract manifest SHA-256 is not frozen")
    if not bool(runtime_report.get("runtime_handoff_ready", False)):
        raise RuntimeError("frozen BC+CBF runtime handoff is not ready")
    if (
        not args.practice_only
        and not bool(runtime_report.get("production_collection_ready", False))
    ):
        raise RuntimeError(
            "study collection is fail-closed: the frozen runtime contract "
            "still has production_collection_ready=false. Complete the "
            "in-HMD practice smoke and explicitly qualify/version the runtime "
            "contract before pilot or production collection."
        )
    live_hmd_qualified = os.environ.get(
        "HRI_LIVE_HMD_QUALIFIED", "0"
    ).strip() == "1"
    if not args.practice_only and not live_hmd_qualified:
        raise RuntimeError(
            "pilot/production collection requires an operator-qualified live "
            "HMD/controller smoke: set HRI_LIVE_HMD_QUALIFIED=1 only after "
            "completing the manual checklist"
        )
    code_provenance = resolve_verified_code_provenance(
        PROJECT_ROOT, str(args.code_commit_sha)
    )
    if (
        not args.practice_only
        and code_provenance["code_commit_verification"]
        != "verified_git_head_clean"
    ):
        raise RuntimeError(
            "pilot/production collection requires a clean Git checkout with "
            "a verifiable HEAD; source-tree-only provenance is practice-only"
        )
    from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp(
        _simulation_config(),
        experience=os.environ.get("ISAAC_SIM_EXPERIENCE", ""),
    )
    env = trace = tracking_guard = recorder = feedback = None
    stop = {"reason": ""}
    collection_committed = {"value": False}

    def request_stop(signum, _frame) -> None:
        if not stop["reason"]:
            stop["reason"] = f"signal_{int(signum)}"

    def current_abort_reason() -> str:
        if stop["reason"]:
            return str(stop["reason"])
        running = getattr(simulation_app, "is_running", None)
        if callable(running) and not running():
            stop["reason"] = "simulation_app_stopped"
        return str(stop["reason"])

    def raise_if_stopped() -> None:
        reason = current_abort_reason()
        if reason:
            raise CollectionAbort(reason)

    previous_signals = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        from v3_chan.explicit_feedback.live_vr import LiveVRTrackingProvider
        from v3_chan.explicit_feedback.xr_feedback import (
            IsaacXRButtonSource,
            RealtimeButtonMapping,
            XRFeedbackController,
        )

        watchdog = TrackingWatchdog(max_pose_age_ms=args.max_pose_age_ms)
        provider = LiveVRTrackingProvider(watchdog)
        policy = FrozenBCPolicy(
            PROJECT_ROOT / POLICY_RELATIVE_PATH, device=args.device
        )
        env = _build_environment(provider, seed=args.seed)
        apply_controlled_hold(env.robot, joint_count=len(env.robot.dof_names))
        env.world.pause()
        provider.setup(env.world, simulation_app)
        env.world.reset()
        env.world.reset()
        apply_controlled_hold(env.robot, joint_count=len(env.robot.dof_names))
        env.world.pause()
        provider.wait_until_ready(
            simulation_app,
            timeout_s=args.tracking_timeout_s,
            stable_frames=args.tracking_stable_frames,
            stop_requested=raise_if_stopped,
        )
        joint_names, arm_joint_indices = validate_environment_contract(
            env,
            expected_max_episode_steps=12_000,
        )
        if len(env.cubes) != 1:
            raise RuntimeError(
                f"single-pick contract requires one cube, got {len(env.cubes)}"
            )
        cbf_config = actual_cbf_config(env._cbf_filter)
        geometry_metadata = env.safety_geometry.metadata()
        protected_links = list(env.safety_geometry.available_links)
        protected_colliders = list(env.safety_geometry.collider_paths)
        if not protected_links or not protected_colliders:
            raise RuntimeError("runtime-discovered distal CBF geometry is empty")
        print(
            f"[OnlineFeedback] CBF protected links: {protected_links}", flush=True
        )
        print(
            f"[OnlineFeedback] CBF protected colliders: {protected_colliders}",
            flush=True,
        )
        proximal_tokens = tuple(
            str(value)
            for value in scenario["protected_crossing"]["proximal_link_tokens"]
        )
        proximal_monitor = ProximalArmProtocolMonitor(
            robot_prim_path=env.safety_geometry.robot_prim_path,
            proximal_link_tokens=proximal_tokens,
            hand_radius_m=env.safety_geometry.thresholds.hand_radius_m,
        )
        if not proximal_monitor.collider_paths:
            raise RuntimeError(
                "runtime-discovered proximal off-protocol geometry is empty"
            )
        missing_proximal_tokens = [
            token
            for token in proximal_tokens
            if not any(
                f"/{token}" in path
                for path in proximal_monitor.collider_paths
            )
        ]
        if missing_proximal_tokens:
            raise RuntimeError(
                "proximal off-protocol geometry is incomplete: "
                + ", ".join(missing_proximal_tokens)
            )
        overlap = sorted(
            set(proximal_monitor.collider_paths).intersection(protected_colliders)
        )
        if overlap:
            raise RuntimeError(
                "distal CBF and proximal off-protocol collider scopes overlap: "
                + ", ".join(overlap)
            )
        print(
            "[OnlineFeedback] proximal off-protocol colliders: "
            f"{list(proximal_monitor.collider_paths)}",
            flush=True,
        )
        metadata = {
            "participant_id": str(args.participant_id),
            "session_id": str(args.session_id),
            "participant_split_key": str(args.participant_id),
            "session_seed": int(args.seed),
            **code_provenance,
            "runtime_handoff_commit": RUNTIME_HANDOFF_COMMIT,
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "live_hmd_qualified": int(live_hmd_qualified),
            "protocol_config_sha256": canonical_sha256(protocol),
            "scenario_config_sha256": canonical_sha256(scenario),
            "cbf_config_sha256": canonical_sha256(cbf_config),
            "cbf_config_json": json.dumps(
                cbf_config, sort_keys=True, separators=(",", ":")
            ),
            "cbf_protected_links_json": json.dumps(
                protected_links, separators=(",", ":")
            ),
            "cbf_protected_colliders_json": json.dumps(
                protected_colliders, separators=(",", ":")
            ),
            "cbf_missing_links_json": json.dumps(
                list(env.safety_geometry.missing_links), separators=(",", ":")
            ),
            "safety_geometry_metadata_json": json.dumps(
                geometry_metadata,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
            "proximal_protocol_monitor": (
                "usd_world_aabb_conservative_read_only_v1"
            ),
            "proximal_collider_paths_json": json.dumps(
                proximal_monitor.collider_paths, separators=(",", ":")
            ),
            "proximal_link_tokens_json": json.dumps(
                proximal_tokens, separators=(",", ":")
            ),
            "proximal_missing_tokens_json": "[]",
            "pilot": int(args.pilot_evaluated_trials is not None),
            "practice_only": int(args.practice_only),
            "smoke_test": int(args.practice_only and len(plans) == 1),
            "physics_dt_s": float(env.physics_dt_s),
            "task_max_episode_steps": int(env.config.max_episode_steps),
            "max_pose_age_ms": float(args.max_pose_age_ms),
            "tracking_stable_frames": int(args.tracking_stable_frames),
            "downstream_action_stage": (
                "gripper_merge_then_articulation_drive_state"
            ),
            "filtered_submitted_velocity_tolerance_rad_s": 1e-6,
            "applied_filtered_validation": (
                "full_vector_units_masks_and_stage_provenance_not_bitwise_equality"
            ),
            "human_feedback_runtime_effect": (
                "logging_only_never_policy_or_cbf_input"
            ),
            "runtime_handoff_ready": int(
                bool(runtime_report.get("runtime_handoff_ready", False))
            ),
            "runtime_production_collection_ready": int(
                bool(
                    runtime_report.get(
                        "production_collection_ready", False
                    )
                )
            ),
        }
        recorder = OnlineExplicitFeedbackRecorder(
            output,
            metadata=metadata,
            joint_names=joint_names,
            arm_joint_indices=arm_joint_indices,
            protocol_config=protocol,
            scenario_config=scenario,
            flush_every_steps=1,
        )
        recorder.write_schedule(plans)
        feedback_mapping = RealtimeButtonMapping.from_mapping(
            protocol["feedback"]["opposite_controller_buttons"]
        )
        feedback = XRFeedbackController(
            realtime_mapping=feedback_mapping,
            questionnaire_mapping=protocol["feedback"]["navigation"],
            questionnaire_timeout_s=float(
                protocol["feedback"]["query_timeout_s"]
            ),
            release_frames=int(protocol["feedback"]["stable_release_frames"]),
        )
        if not feedback.attach_xr_ui():
            raise CollectionAbort("in-HMD XRSceneView feedback UI is unavailable")
        button_source = IsaacXRButtonSource(avatar=provider.avatar)
        _wait_for_controller_release(
            source=button_source,
            feedback=feedback,
            provider=provider,
            simulation_app=simulation_app,
            env=env,
            timeout_s=args.tracking_timeout_s,
            stop_requested=raise_if_stopped,
        )
        _set_instruction(
            feedback,
            "BC-only layout precheck 중입니다. 양손을 staging 위치에 유지하세요.",
        )
        initial_scene, restoration, layout_id, precheck_index = (
            _run_bc_only_precheck(
                env=env,
                policy=policy,
                provider=provider,
                recorder=recorder,
                protocol=protocol,
                scenario=scenario,
                seed=args.seed,
                simulation_app=simulation_app,
                proximal_monitor=proximal_monitor,
                arm_joint_indices=arm_joint_indices,
                stop_requested=raise_if_stopped,
            )
        )
        trace = ActionTraceRecorder(
            robot=env.robot, controller=env.controller, cbf=env._cbf_filter
        ).install()
        tracking_guard = PreApplyTrackingGuard(
            robot=env.robot,
            provider=provider,
            watchdog=watchdog,
            trace=trace,
            abort_reason_fn=lambda: stop["reason"],
        ).install()
        cue_visuals = CrossingCueVisuals(
            env.world,
            corridor_radius_m=float(scenario["crossing"]["corridor_radius_m"]),
            start_radius_m=float(scenario["crossing"]["start_radius_m"]),
            end_radius_m=float(scenario["crossing"]["end_radius_m"]),
        )
        for spec, plan in zip(specs, plans):
            _run_trial(
                spec=spec,
                plan=plan,
                env=env,
                policy=policy,
                provider=provider,
                watchdog=watchdog,
                trace=trace,
                recorder=recorder,
                feedback=feedback,
                button_source=button_source,
                cue_visuals=cue_visuals,
                proximal_monitor=proximal_monitor,
                protocol=protocol,
                scenario=scenario,
                restoration=restoration,
                initial_scene=initial_scene,
                layout_id=layout_id,
                layout_precheck_index=precheck_index,
                arm_joint_indices=arm_joint_indices,
                simulation_app=simulation_app,
                args=args,
                stop_requested=raise_if_stopped,
            )
        raise_if_stopped()
        apply_controlled_hold(env.robot, joint_count=len(joint_names))
        env.world.pause()
        partial = recorder.seal()
        from v3_chan.explicit_feedback.online_validator import (
            validate_online_collection,
        )

        report = validate_online_collection(
            partial, allow_partial_path=True, raise_on_error=False
        )
        if not report.valid:
            detail = "; ".join(
                f"[{issue.code}] {issue.path}: {issue.message}"
                for issue in report.issues[:16]
            )
            raise CollectionAbort(
                "sealed collection failed validation: " + detail
            )
        final = _commit_validated_with_signal_barrier(
            recorder=recorder,
            abort_reason_fn=current_abort_reason,
            commit_state=collection_committed,
        )
        print(f"[OnlineFeedback] validated collection: {final}", flush=True)
        return final
    except BaseException as error:
        if trace is not None and trace.step_active:
            trace.abort_step()
        if env is not None:
            try:
                apply_controlled_hold(
                    env.robot, joint_count=len(env.robot.dof_names)
                )
                env.world.pause()
            except Exception:
                pass
        if recorder is not None and not getattr(recorder, "_closed", True):
            partial = recorder.abort(f"{type(error).__name__}:{error}")
            print(
                f"[OnlineFeedback] partial collection preserved: {partial}",
                flush=True,
            )
        raise
    finally:
        if feedback is not None:
            feedback.close()
        if tracking_guard is not None:
            try:
                tracking_guard.uninstall()
            except Exception:
                pass
        if trace is not None:
            try:
                trace.uninstall()
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            simulation_app.close()
        except Exception:
            pass
        for signum, previous in previous_signals.items():
            signal.signal(signum, previous)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _finite_or_zero(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _normalized_task_failure_reason(
    info: Mapping[str, Any],
    *,
    strict_state: Mapping[str, Any],
    task_success: bool,
    reset_after_query: bool,
    max_episode_steps: int,
) -> str:
    """Return an actual failure reason; success is never encoded as failure."""

    if bool(task_success):
        return ""
    terminal = str(info.get("task_terminal_reason", ""))
    if terminal == "success":
        terminal = ""
    return str(
        strict_state.get("failure_reason", "")
        or terminal
        or (
            "max_episode_steps"
            if int(info.get("step", 0)) >= int(max_episode_steps)
            else ("reset_after_query" if reset_after_query else "")
        )
    )


simulation_app = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        protocol, scenario = _load_configs(args)
        specs, plans = _build_schedule(args)
        if args.dry_run:
            payload = {
                "schema_version": protocol["schema_version"],
                "participant_id": args.participant_id,
                "session_id": args.session_id,
                "protocol_config_sha256": canonical_sha256(protocol),
                "scenario_config_sha256": canonical_sha256(scenario),
                "trial_count": len(plans),
                "practice_count": sum(plan.practice for plan in plans),
                "evaluated_count": sum(not plan.practice for plan in plans),
                "anchor_count": sum(plan.anchor_repeat for plan in plans),
                "pilot": args.pilot_evaluated_trials is not None,
                "haptics_enabled": False,
                "trials": [plan.as_dict() for plan in plans],
            }
            print(
                json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, indent=2
                )
            )
            return 0
        _run_collection(args, protocol, scenario, specs, plans)
        return 0
    except KeyboardInterrupt:
        print(
            "[OnlineFeedback] interrupted; no final study file was produced",
            flush=True,
        )
        return 130
    except Exception as error:
        print(
            f"[OnlineFeedback] ABORTED: {type(error).__name__}: {error}",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
