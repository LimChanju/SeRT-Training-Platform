"""Pure-Python contract tests for the A/C VR feedback collector.

These tests intentionally avoid Isaac Sim, Torch, HDF5, and a live XR runtime.
They protect the study design and controller-input state machines that must be
stable before an operator starts an in-HMD practice trial.
"""

from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from v3_chan.collect_ac_selective_smoothing_feedback import (
    CollectionAbort,
    DIRECT_LIVE_CORRIDOR_SEMANTICS,
    DIRECT_LIVE_LAYOUT_SEMANTICS,
    _cached_pose_age_only,
    _capture_protocol_hold_pose,
    _initialize_direct_live_scene,
    _prepare_current_control_observation,
    _read_required_controller_inputs,
    _require_post_step_tracking,
    _refresh_expired_control_observation,
    _wait_for_controller_release,
    _wait_for_tracking_and_clear,
)
import numpy as np

from v3_chan.ac_feedback.config import (
    DEFAULT_CONFIG,
    canonical_sha256,
    load_config,
    validate_config,
)
from v3_chan.ac_feedback.online_protocol import (
    CrossingPathMonitor,
    resolve_actual_crossing_direction,
)
from v3_chan.ac_feedback.online_scenario import CrossingCueVisuals
from v3_chan.ac_feedback.online_recorder import OnlineExplicitFeedbackRecorder
from v3_chan.ac_feedback.recorder import DatasetSpec
from v3_chan.ac_feedback.online_schema import (
    ACTUAL_CROSSING_DIRECTIONS,
    Q1_ID as ONLINE_Q1_ID,
    RealtimeMarkerRecord,
)
from v3_chan.ac_feedback.study import (
    ResponsePhase,
    ResponsePhaseConfig,
    SafetyResponsePhaseDetector,
    TASK_PHASES,
    TrialLifecycle,
    TrialState,
    build_participant_schedule,
    validate_decision_context,
)
from v3_chan.ac_feedback.xr_feedback import (
    ControllerButtonSnapshot,
    ControllerInputUnavailable,
    ControllerInputWatchdog,
    EmergencyAbortMonitor,
    FeedbackClocks,
    LIKERT_QUESTIONS_KO,
    Q1_RESPONSE_CODES,
    QuestionnaireButtonMapping,
    QuestionnaireFSM,
    REALTIME_BEHAVIOR_ANOMALY,
    REALTIME_SAFETY_CONCERN,
    REJECTION_REASON_IDS,
    RealtimeButtonMapping,
    RealtimeMarkerInput,
    XRFeedbackController,
    XRSceneViewTextRenderer,
)
from v3_chan.ac_feedback.live_vr import (
    _configure_xr_profile_settings,
    _xr_token_text,
    LiveVRTrackingProvider,
    verify_feedback_only_action_map,
    xr_prestartup_args,
)
from v3_chan.ac_feedback.tracking import DEFAULT_ALLOWED_SOURCES, TrackingDropout
from v3_chan.ac_feedback.tracking_guard import PreApplyTrackingGuard
from v3_chan.rl.pick_place_env import (
    IsaacPickPlaceEnv,
    _validated_controller_pose_override,
)
from v3_chan.ac_feedback.validator import (
    _Validator,
    _missing_resume_outcome_matches,
)


def _maximum_run(values: list[str]) -> int:
    longest = 0
    current = 0
    previous = None
    for value in values:
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
        previous = value
    return longest


def _clock(index: int) -> FeedbackClocks:
    return FeedbackClocks(
        sim_time_s=index / 10.0,
        monotonic_ns=1_000_000_000 + index * 100_000_000,
        unix_ns=2_000_000_000 + index * 100_000_000,
        control_step=index,
    )


class ScheduleContractTests(unittest.TestCase):
    def test_minimal_schedule_is_deterministic_and_balanced(self) -> None:
        schedule = build_participant_schedule(
            "pilot_p01", session_id="session_01", seed=11,
            mode="minimal_pilot",
        )
        repeated = build_participant_schedule(
            "pilot_p01", session_id="session_01", seed=11,
            mode="minimal_pilot",
        )

        self.assertEqual(schedule.trial_order, repeated.trial_order)
        self.assertEqual(schedule.schedule_seed, repeated.schedule_seed)
        self.assertEqual(len(schedule.practice_trials), 4)
        self.assertEqual(len(schedule.evaluated_trials), 16)
        self.assertTrue(all(trial.analysis_exclude for trial in schedule.practice_trials))
        core = list(schedule.evaluated_trials)
        self.assertEqual(
            Counter(trial.condition_id for trial in core),
            Counter({"A_reactive": 8, "C_smooth": 8}),
        )
        for phase in TASK_PHASES:
            phase_trials = [trial for trial in core if trial.task_phase == phase]
            self.assertEqual(
                Counter(trial.condition_id for trial in phase_trials),
                Counter({"A_reactive": 2, "C_smooth": 2}),
            )
        for severity in ("shallow", "threat"):
            severity_trials = [trial for trial in core if trial.severity == severity]
            self.assertEqual(
                Counter(trial.condition_id for trial in severity_trials),
                Counter({"A_reactive": 4, "C_smooth": 4}),
            )
        self.assertEqual(
            Counter(trial.crossing_direction for trial in core),
            Counter({"left_to_right": 8, "right_to_left": 8}),
        )
        self.assertEqual(
            Counter(trial.crossing_speed for trial in core),
            Counter({"slow": 8, "fast": 8}),
        )
        self.assertLessEqual(_maximum_run([trial.condition_id for trial in core]), 2)
        self.assertEqual(
            [trial.order_index for trial in schedule], list(range(len(schedule)))
        )

    def test_first_condition_is_counterbalanced_across_participant_ids(self) -> None:
        first_conditions = {
            build_participant_schedule(
                f"pilot_p{index:02d}",
                session_id="session_01",
                seed=11,
                mode="minimal_pilot",
            ).evaluated_trials[0].condition_id
            for index in range(1, 33)
        }
        self.assertEqual(first_conditions, {"A_reactive", "C_smooth"})

    def test_anchor_mode_adds_two_contexts_repeated_under_a_and_c(self) -> None:
        schedule = build_participant_schedule(
            "pilot_p01", session_id="session_anchors", seed=11,
            mode="pilot_with_anchors",
        )
        anchors = [trial for trial in schedule.evaluated_trials if trial.anchor_repeat]
        self.assertEqual(len(schedule.practice_trials), 4)
        self.assertEqual(len(schedule.evaluated_trials), 20)
        self.assertEqual(len(anchors), 4)
        self.assertEqual(len({trial.anchor_context_id for trial in anchors}), 2)
        for context_id in {trial.anchor_context_id for trial in anchors}:
            context = [trial for trial in anchors if trial.anchor_context_id == context_id]
            self.assertEqual(
                Counter(trial.condition_id for trial in context),
                Counter({"A_reactive": 1, "C_smooth": 1}),
            )


class TrialLifecycleTests(unittest.TestCase):
    def test_exact_seventeen_state_lifecycle_and_cardinality(self) -> None:
        lifecycle = TrialLifecycle("A_reactive")
        for state in list(TrialState)[1:]:
            lifecycle.advance(state)
            if state is TrialState.EXECUTE_SINGLE_CROSSING:
                lifecycle.record_crossing()
            elif state is TrialState.SHOW_MANDATORY_FEEDBACK_UI:
                lifecycle.record_query()
            elif state is TrialState.SAVE_FEEDBACK:
                lifecycle.record_feedback_saved()
        lifecycle.validate_complete()
        self.assertEqual(lifecycle.history, [state.value for state in TrialState])

    def test_illegal_skip_and_duplicate_crossing_fail_closed(self) -> None:
        lifecycle = TrialLifecycle("C_smooth")
        with self.assertRaisesRegex(RuntimeError, "illegal trial transition"):
            lifecycle.advance(TrialState.START_FROZEN_BC)
        for state in list(TrialState)[1:6]:
            lifecycle.advance(state)
        lifecycle.record_crossing()
        with self.assertRaisesRegex(RuntimeError, "more than one intended crossing"):
            lifecycle.record_crossing()

    def test_missing_resume_accepts_terminal_on_final_query_row(self) -> None:
        states = [
            TrialState.PAUSE_NOMINAL_TASK_PROGRESSION.value,
            TrialState.SHOW_MANDATORY_FEEDBACK_UI.value,
            TrialState.SHOW_MANDATORY_FEEDBACK_UI.value,
        ]
        self.assertTrue(
            _missing_resume_outcome_matches(
                states=states,
                step_success=[0, 0, 0],
                step_terminated=[0, 0, 1],
                step_truncated=[0, 0, 0],
                terminal_reasons=["", "", "object_drop"],
                summary_success=False,
                completion_observed=True,
                failure_reason="object_drop",
            )
        )
        self.assertTrue(
            _missing_resume_outcome_matches(
                states=states,
                step_success=[0, 0, 1],
                step_terminated=[0, 0, 0],
                step_truncated=[0, 0, 0],
                terminal_reasons=["", "", ""],
                summary_success=True,
                completion_observed=True,
                failure_reason="",
            )
        )

    def test_missing_resume_rejects_late_or_misbound_terminal_evidence(self) -> None:
        states = [
            TrialState.SHOW_MANDATORY_FEEDBACK_UI.value,
            TrialState.SAVE_FEEDBACK.value,
        ]
        common = {
            "states": states,
            "step_success": [0, 0],
            "step_terminated": [0, 1],
            "step_truncated": [0, 0],
            "terminal_reasons": ["", "object_drop"],
            "summary_success": False,
            "completion_observed": True,
            "failure_reason": "object_drop",
        }
        self.assertFalse(_missing_resume_outcome_matches(**common))

        first_reason_mismatch = {
            **common,
            "states": [
                TrialState.SHOW_MANDATORY_FEEDBACK_UI.value,
                TrialState.SHOW_MANDATORY_FEEDBACK_UI.value,
            ],
            "step_terminated": [1, 1],
            "terminal_reasons": ["grasp_lost", "object_drop"],
        }
        self.assertFalse(
            _missing_resume_outcome_matches(**first_reason_mismatch)
        )
        self.assertTrue(
            _missing_resume_outcome_matches(
                **{
                    **first_reason_mismatch,
                    "failure_reason": "grasp_lost",
                }
            )
        )


class ResponsePhaseDetectorTests(unittest.TestCase):
    def test_onset_tail_recovery_and_stable_resumption_are_separate(self) -> None:
        detector = SafetyResponsePhaseDetector(ResponsePhaseConfig())

        low = detector.update(
            control_step=0, simulation_time_s=0.00,
            cbf_constraint_active=True, intervention_norm_rad_s=0.049,
            smooth_tail_active=False, recovery_active=False, bc_resumed=False,
        )
        self.assertEqual(low.phase, ResponsePhase.PRE_RESPONSE)
        candidate = detector.update(
            control_step=1, simulation_time_s=0.01,
            cbf_constraint_active=True, intervention_norm_rad_s=0.05,
            smooth_tail_active=False, recovery_active=False, bc_resumed=False,
        )
        self.assertTrue(candidate.onset_candidate_now)
        detector.update(
            control_step=2, simulation_time_s=0.02,
            cbf_constraint_active=True, intervention_norm_rad_s=0.06,
            smooth_tail_active=False, recovery_active=False, bc_resumed=False,
        )
        confirmed = detector.update(
            control_step=3, simulation_time_s=0.03,
            cbf_constraint_active=True, intervention_norm_rad_s=0.07,
            smooth_tail_active=False, recovery_active=False, bc_resumed=False,
        )
        self.assertTrue(confirmed.onset_confirmed_now)
        self.assertEqual(confirmed.onset_candidate_step, 1)
        self.assertEqual(confirmed.onset_confirmed_step, 3)
        self.assertEqual(confirmed.phase, ResponsePhase.CBF_ACTIVE)

        tail = detector.update(
            control_step=4, simulation_time_s=0.04,
            cbf_constraint_active=False, intervention_norm_rad_s=0.03,
            smooth_tail_active=True, recovery_active=False, bc_resumed=False,
        )
        self.assertEqual(tail.phase, ResponsePhase.SMOOTH_TAIL)
        recovery = detector.update(
            control_step=5, simulation_time_s=0.05,
            cbf_constraint_active=False, intervention_norm_rad_s=0.02,
            smooth_tail_active=False, recovery_active=True, bc_resumed=False,
        )
        self.assertEqual(recovery.phase, ResponsePhase.RECOVERY_ACTIVE)
        self.assertEqual(recovery.recovery_onset_step, 5)
        resumed = detector.update(
            control_step=6, simulation_time_s=0.10,
            cbf_constraint_active=False, intervention_norm_rad_s=0.01,
            smooth_tail_active=False, recovery_active=False, bc_resumed=True,
        )
        self.assertEqual(resumed.phase, ResponsePhase.BC_RESUMED)
        not_stable = detector.update(
            control_step=7, simulation_time_s=0.59,
            cbf_constraint_active=False, intervention_norm_rad_s=0.0,
            smooth_tail_active=False, recovery_active=False, bc_resumed=True,
        )
        self.assertFalse(not_stable.stable_resumption_now)
        stable = detector.update(
            control_step=8, simulation_time_s=0.61,
            cbf_constraint_active=False, intervention_norm_rad_s=0.0,
            smooth_tail_active=False, recovery_active=False, bc_resumed=True,
        )
        self.assertTrue(stable.stable_resumption_now)
        self.assertEqual(stable.phase, ResponsePhase.STABLE_TASK_RESUMPTION)

        terminal = detector.update(
            control_step=99, simulation_time_s=2.0,
            cbf_constraint_active=True, intervention_norm_rad_s=0.2,
            smooth_tail_active=False, recovery_active=False, bc_resumed=False,
        )
        self.assertEqual(terminal.phase, ResponsePhase.STABLE_TASK_RESUMPTION)
        self.assertFalse(terminal.stable_resumption_now)

    def test_interrupted_candidate_requires_three_new_consecutive_frames(self) -> None:
        detector = SafetyResponsePhaseDetector()
        common = dict(
            smooth_tail_active=False, recovery_active=False, bc_resumed=False,
        )
        first = detector.update(
            control_step=1, simulation_time_s=0.01,
            cbf_constraint_active=True, intervention_norm_rad_s=0.06, **common,
        )
        self.assertTrue(first.onset_candidate_now)
        detector.update(
            control_step=2, simulation_time_s=0.02,
            cbf_constraint_active=False, intervention_norm_rad_s=0.0, **common,
        )
        second = detector.update(
            control_step=3, simulation_time_s=0.03,
            cbf_constraint_active=True, intervention_norm_rad_s=0.06, **common,
        )
        self.assertTrue(second.onset_candidate_now)
        self.assertEqual(second.onset_candidate_step, 3)
        detector.update(
            control_step=4, simulation_time_s=0.04,
            cbf_constraint_active=True, intervention_norm_rad_s=0.06, **common,
        )
        third = detector.update(
            control_step=5, simulation_time_s=0.05,
            cbf_constraint_active=True, intervention_norm_rad_s=0.06, **common,
        )
        self.assertTrue(third.onset_confirmed_now)

    def test_recovery_edges_ignore_phase_precedence_and_clamp_to_response(self) -> None:
        detector = SafetyResponsePhaseDetector()
        pre = detector.update(
            control_step=0, simulation_time_s=0.00,
            cbf_constraint_active=False, intervention_norm_rad_s=0.0,
            smooth_tail_active=False, recovery_active=True, bc_resumed=False,
        )
        self.assertTrue(pre.recovery_onset_now)
        self.assertEqual(pre.recovery_onset_step, -1)

        updates = []
        for step in (1, 2, 3):
            updates.append(detector.update(
                control_step=step, simulation_time_s=step / 100.0,
                cbf_constraint_active=True, intervention_norm_rad_s=0.06,
                smooth_tail_active=False, recovery_active=True,
                bc_resumed=False,
            ))
        confirmed = updates[-1]
        self.assertEqual(confirmed.phase, ResponsePhase.CBF_ACTIVE)
        self.assertEqual(confirmed.recovery_onset_step, 1)
        self.assertEqual(confirmed.recovery_onset_simulation_time_s, 0.01)

        tail = detector.update(
            control_step=4, simulation_time_s=0.04,
            cbf_constraint_active=False, intervention_norm_rad_s=0.02,
            smooth_tail_active=True, recovery_active=True, bc_resumed=False,
        )
        self.assertEqual(tail.phase, ResponsePhase.SMOOTH_TAIL)
        ended = detector.update(
            control_step=5, simulation_time_s=0.05,
            cbf_constraint_active=False, intervention_norm_rad_s=0.0,
            smooth_tail_active=False, recovery_active=False, bc_resumed=True,
        )
        self.assertTrue(ended.recovery_end_now)
        self.assertEqual(ended.recovery_end_step_exclusive, 5)
        self.assertEqual(ended.recovery_end_simulation_time_s, 0.05)


class _ArrayDataset:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)

    def __getitem__(self, key: object) -> np.ndarray:
        if key != ():
            raise KeyError(key)
        return self.values

    def asstr(self) -> "_ArrayDataset":
        return self


class _ArrayGroup(dict[str, _ArrayDataset]):
    def __init__(self, values: dict[str, object], attrs: dict[str, object]) -> None:
        super().__init__(
            (name, _ArrayDataset(value)) for name, value in values.items()
        )
        self.attrs = attrs


class QueryHoldValidationTests(unittest.TestCase):
    @staticmethod
    def _validator() -> _Validator:
        validator = object.__new__(_Validator)
        validator.issues = []
        return validator

    @staticmethod
    def _group() -> _ArrayGroup:
        states = [
            TrialState.WAIT_FOR_TARGET_TASK_PHASE.value,
            TrialState.SHOW_HAND_CROSSING_CUE.value,
            TrialState.TRACK_SAFETY_RESPONSE_EPISODE.value,
            TrialState.SHOW_RETURN_HAND_TO_NEUTRAL_CUE.value,
            TrialState.PAUSE_NOMINAL_TASK_PROGRESSION.value,
            TrialState.SHOW_MANDATORY_FEEDBACK_UI.value,
            TrialState.RESUME_TASK_TO_COMPLETION_OR_TERMINAL.value,
        ]
        count = len(states)
        holds = np.asarray([False, True, False, False, True, True, False])
        return _ArrayGroup(
            {
                "control/trial_state": states,
                "control/control_mode": np.where(
                    holds, "protocol_hold", "nominal_bc"
                ),
                "task/phase_advance_enabled": np.where(holds, 0, 1),
                "actions/controller_task_action": np.zeros((count, 5)),
                "actions/controller_task_action_reason": np.where(
                    holds,
                    "current_ee_protocol_hold",
                    "frozen_bc_policy",
                ),
                "recovery/control_authority": np.zeros(count),
                "actions/action_pipeline_complete": np.ones(count),
                "cbf/intervention_available": np.ones(count),
                "safety/geometry_valid": np.ones(count),
                "recovery/stable_task_resumption": np.asarray(
                    [0, 0, 1, 1, 1, 1, 1]
                ),
            },
            {
                "response_onset_confirmed_step": 1,
                "stable_task_resumption_step": 2,
            },
        )

    def test_query_hold_requires_zero_sentinel_and_complete_cbf_evidence(self) -> None:
        group = self._group()
        validator = self._validator()
        validator._validate_fsm_rows(group, "/trials/t1")
        self.assertEqual(validator.issues, [])

        for field, bad_value in (
            ("actions/controller_task_action", (4, 0, 0.1)),
            ("recovery/control_authority", (4, 1)),
            ("cbf/intervention_available", (5, 0)),
            ("safety/geometry_valid", (5, 0)),
        ):
            with self.subTest(field=field):
                mutated = self._group()
                index_values = bad_value
                if len(index_values) == 3:
                    row, column, value = index_values
                    mutated[field].values[row, column] = value
                else:
                    row, value = index_values
                    mutated[field].values[row] = value
                validator = self._validator()
                validator._validate_fsm_rows(mutated, "/trials/t1")
                self.assertTrue(
                    any(issue.code == "trial.query_hold" for issue in validator.issues)
                )


class CrossingDirectionAuditTests(unittest.TestCase):
    def test_protocol_hold_captures_the_motion_policy_control_frame(self) -> None:
        active_positions = np.asarray([0.1, -0.2, 0.3], dtype=np.float64)
        expected_position = np.asarray([0.4, 0.1, 1.2], dtype=np.float64)
        expected_rotation = np.eye(3, dtype=np.float64)
        expected_orientation = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )

        class ActiveSubset:
            @staticmethod
            def get_joint_positions():
                return active_positions.copy()

        class ArticulationPolicy:
            @staticmethod
            def get_active_joints_subset():
                return ActiveSubset()

        class MotionPolicy:
            @staticmethod
            def get_end_effector_pose(joint_positions):
                np.testing.assert_array_equal(joint_positions, active_positions)
                return expected_position.copy(), expected_rotation.copy()

        class Controller:
            @staticmethod
            def get_articulation_motion_policy():
                return ArticulationPolicy()

            @staticmethod
            def get_motion_policy():
                return MotionPolicy()

        env = object.__new__(IsaacPickPlaceEnv)
        env.controller = Controller()
        env._rot_matrix_to_quat = lambda value: (
            expected_orientation.copy()
            if np.array_equal(value, expected_rotation)
            else np.zeros(4, dtype=np.float64)
        )
        position, orientation = env.capture_motion_policy_end_effector_pose()
        np.testing.assert_array_equal(position, expected_position)
        np.testing.assert_array_equal(orientation, expected_orientation)

        captured_position, captured_orientation = _capture_protocol_hold_pose(env)
        np.testing.assert_array_equal(captured_position, expected_position)
        np.testing.assert_array_equal(
            captured_orientation, expected_orientation
        )

    def test_protocol_hold_pose_override_is_strict_and_normalized(self) -> None:
        position, orientation = _validated_controller_pose_override(
            (
                np.asarray([0.4, 0.1, 1.2]),
                np.asarray([2.0, 0.0, 0.0, 0.0]),
            )
        )
        np.testing.assert_array_equal(position, [0.4, 0.1, 1.2])
        np.testing.assert_array_equal(orientation, [1.0, 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "nonzero"):
            _validated_controller_pose_override(
                (np.zeros(3), np.zeros(4))
            )

    def test_protocol_hold_routes_latched_pose_through_cbf_and_apply(self) -> None:
        class Action:
            def __init__(self, value: float) -> None:
                self.joint_positions = np.full(9, value, dtype=np.float64)
                self.joint_velocities = np.full(9, value, dtype=np.float64)
                self.joint_efforts = None

        nominal = Action(1.0)
        filtered = Action(2.0)
        target_position = np.asarray([0.4, 0.1, 1.2], dtype=np.float64)
        target_orientation = np.asarray(
            [0.5, 0.5, 0.5, 0.5], dtype=np.float64
        )
        calls: dict[str, object] = {}

        class Robot:
            dof_names = tuple(f"joint_{index}" for index in range(9))

            @staticmethod
            def get_joint_positions():
                return np.zeros(9, dtype=np.float64)

            @staticmethod
            def get_joint_velocities():
                return np.zeros(9, dtype=np.float64)

            @staticmethod
            def apply_action(action) -> None:
                calls["applied"] = action

        class Controller:
            @staticmethod
            def forward(
                *,
                target_end_effector_position,
                target_end_effector_orientation,
            ):
                calls["controller_target"] = (
                    np.asarray(target_end_effector_position).copy(),
                    np.asarray(target_end_effector_orientation).copy(),
                )
                return nominal

        class CBF:
            config = SimpleNamespace(correction_smoothness_weight=0.0)

            @staticmethod
            def filter_action(**kwargs):
                calls["cbf_arm_action"] = kwargs["arm_action"]
                calls["cbf_context"] = kwargs["task_progress_context"]
                return filtered, SimpleNamespace(intervention_norm_radps=0.0)

            @staticmethod
            def notify_action_committed(committed) -> None:
                calls["committed"] = committed

        class World:
            current_time = 1.0

            @staticmethod
            def step(*, render) -> None:
                calls["world_render"] = render

        inactive_recovery = SimpleNamespace(
            target_position_m=None,
            timed_out=False,
            handoff=False,
        )
        env = object.__new__(IsaacPickPlaceEnv)
        env._last_obs = {"current": np.zeros(1, dtype=np.float32)}
        env.robot = Robot()
        env.controller = Controller()
        env._curobo_controller = None
        env._cbf_filter = CBF()
        env._physical_safety_mode = "cbf"
        env._rmpflow_valid_hand_count = 2
        env._last_safety_result = SimpleNamespace(distance_gate=0.0)
        env._last_dynamic_safety_sample = object()
        env._human_replay_aux_state = {}
        env.safety_geometry = object()
        env.physics_dt_s = 1.0 / 60.0
        env.step_count = 0
        env.world = World()
        env.config = SimpleNamespace(
            render=False,
            max_episode_steps=100,
            success_dist=0.06,
            cbf_objective_mode="joint_nominal",
            reward_weights=None,
            reward_version="test",
            minimal_reward_weights=None,
            isaac_franka_dense_reward_weights=None,
        )
        env._prepare_state_aware_recovery_control = (
            lambda _obs: inactive_recovery
        )
        env._cbf_task_progress_context = lambda **kwargs: {
            "target": np.asarray(kwargs["target_pos"]).copy()
        }
        env._guard_strict_release_for_current_cbf = lambda command: command

        def merge(arm_action, gripper_command):
            calls["merged"] = (arm_action, gripper_command)
            return arm_action

        env._merge_gripper_action = merge
        env._strict_task_semantics_enabled = lambda: False
        next_obs = {"next": np.ones(1, dtype=np.float32)}
        env._build_obs = lambda: next_obs
        env._control_task_phase = lambda *_args, **_kwargs: {
            "advanced": False,
            "phase_changed": False,
            "paused": True,
            "reentry": False,
        }
        env._is_success = lambda _obs: False
        env._pseudo_errp_result = lambda *_args, **_kwargs: SimpleNamespace(
            feedback=0.0
        )
        env._info = lambda *_args, **_kwargs: {"step": 1}
        env._format_obs = lambda value: value

        with patch(
            "v3_chan.rl.pick_place_env.compute_reward",
            return_value=SimpleNamespace(total=0.0, components={}),
        ):
            observed, _reward, _terminated, _truncated, info = env.step(
                np.zeros(5, dtype=np.float32),
                advance_task_phase=False,
                controller_end_effector_pose_override=(
                    target_position,
                    target_orientation,
                ),
            )

        sent_position, sent_orientation = calls["controller_target"]
        np.testing.assert_array_equal(sent_position, target_position)
        np.testing.assert_array_equal(sent_orientation, target_orientation)
        self.assertIs(calls["cbf_arm_action"], nominal)
        np.testing.assert_array_equal(
            calls["cbf_context"]["target"], target_position
        )
        self.assertEqual(calls["merged"], (filtered, None))
        self.assertIs(calls["applied"], filtered)
        self.assertIs(calls["committed"], True)
        self.assertIs(calls["world_render"], False)
        self.assertIs(observed, next_obs)
        self.assertTrue(info["task_phase_paused"])

    def test_protocol_hold_cannot_replace_active_recovery(self) -> None:
        env = object.__new__(IsaacPickPlaceEnv)
        env._last_obs = {}
        env.robot = SimpleNamespace(
            get_joint_positions=lambda: np.zeros(9),
            get_joint_velocities=lambda: np.zeros(9),
        )
        env._curobo_controller = None
        env._prepare_state_aware_recovery_control = lambda _obs: SimpleNamespace(
            target_position_m=np.asarray([0.4, 0.1, 1.2]),
            timed_out=False,
            handoff=False,
        )
        with self.assertRaisesRegex(RuntimeError, "active Recovery"):
            env.step(
                np.zeros(5, dtype=np.float32),
                advance_task_phase=False,
                controller_end_effector_pose_override=(
                    np.asarray([0.4, 0.1, 1.2]),
                    np.asarray([1.0, 0.0, 0.0, 0.0]),
                ),
            )

    def test_protocol_hold_rejects_curobo_frame_mismatch(self) -> None:
        env = object.__new__(IsaacPickPlaceEnv)
        env._curobo_controller = object()
        with self.assertRaisesRegex(RuntimeError, "cuRobo"):
            env.capture_motion_policy_end_effector_pose()

    def test_crossing_cue_uses_valid_usd_prim_identifiers(self) -> None:
        created_paths: list[str] = []

        class VisualSphere:
            def __init__(self, **kwargs) -> None:
                created_paths.append(str(kwargs["prim_path"]))

        class Scene:
            @staticmethod
            def add(value):
                return value

        objects_module = types.ModuleType("omni.isaac.core.objects")
        objects_module.VisualSphere = VisualSphere
        core_module = types.ModuleType("omni.isaac.core")
        isaac_module = types.ModuleType("omni.isaac")
        omni_module = types.ModuleType("omni")
        with patch.dict(
            sys.modules,
            {
                "omni": omni_module,
                "omni.isaac": isaac_module,
                "omni.isaac.core": core_module,
                "omni.isaac.core.objects": objects_module,
            },
        ):
            CrossingCueVisuals(
                SimpleNamespace(scene=Scene()),
                sample_count=5,
                corridor_radius_m=0.01,
                start_radius_m=0.02,
                end_radius_m=0.02,
            )

        corridor_paths = [
            path for path in created_paths if "/Corridor/" in path
        ]
        self.assertEqual(
            corridor_paths,
            [
                f"/World/OnlineFeedback/Corridor/seg_{index:02d}"
                for index in range(5)
            ],
        )
        self.assertTrue(
            all(
                segment and (segment[0].isalpha() or segment[0] == "_")
                for path in created_paths
                for segment in path.split("/")
                if segment
            )
        )

    @staticmethod
    def _monitor() -> CrossingPathMonitor:
        return CrossingPathMonitor(
            start_position_world=(0.0, 0.0, 0.0),
            end_position_world=(1.0, 0.0, 0.0),
            path_deviation_threshold_m=0.05,
        )

    def test_first_completed_traversal_preserves_its_signed_direction(self) -> None:
        forward = self._monitor()
        forward.observe(
            position_world=(0.0, 0.0, 0.0), simulation_time_s=0.0
        )
        forward.observe(
            position_world=(1.0, 0.0, 0.0), simulation_time_s=1.0
        )
        forward_metrics = forward.finalize()
        self.assertEqual(forward_metrics.first_completed_traversal_sign, 1)
        self.assertEqual(
            resolve_actual_crossing_direction(
                "left_to_right",
                forward_metrics.first_completed_traversal_sign,
            ),
            "left_to_right",
        )

        reverse = self._monitor()
        reverse.observe(
            position_world=(1.0, 0.0, 0.0), simulation_time_s=0.0
        )
        reverse.observe(
            position_world=(0.0, 0.0, 0.0), simulation_time_s=1.0
        )
        reverse_metrics = reverse.finalize()
        self.assertEqual(reverse_metrics.first_completed_traversal_sign, -1)
        self.assertEqual(
            resolve_actual_crossing_direction(
                "left_to_right",
                reverse_metrics.first_completed_traversal_sign,
            ),
            "right_to_left",
        )

    def test_incomplete_traversal_uses_explicit_not_observed_sentinel(self) -> None:
        incomplete = self._monitor().finalize()
        self.assertEqual(incomplete.first_completed_traversal_sign, 0)
        self.assertEqual(
            resolve_actual_crossing_direction(
                "right_to_left", incomplete.first_completed_traversal_sign
            ),
            "not_observed",
        )
        self.assertIn("not_observed", ACTUAL_CROSSING_DIRECTIONS)


class FrozenConfigTests(unittest.TestCase):
    def test_default_config_matches_frozen_ac_and_feedback_contract(self) -> None:
        config = load_config()
        validate_config(config)
        self.assertEqual(
            config["study"]["available_modes"],
            {
                "minimal_pilot": {
                    "evaluated_trials": 16, "anchor_repeat_trials": 0,
                },
                "pilot_with_anchors": {
                    "evaluated_trials": 20, "anchor_repeat_trials": 4,
                },
            },
        )
        self.assertEqual(
            config["conditions"],
            {
                "A_reactive": {"objective_mode": "joint_nominal", "lambda_s": 0.0},
                "C_smooth": {"objective_mode": "smooth_intervention", "lambda_s": 4.0},
            },
        )
        self.assertFalse(config["feedback"]["haptics_enabled"])
        self.assertFalse(config["study"]["feedback_changes_schedule"])
        self.assertFalse(config["policy"]["online_updates"])
        self.assertFalse(config["shared_cbf"]["parameter_adaptation"])
        self.assertEqual(
            config["layout_precheck"]["staging_dwell_wall_s"], 5.0
        )
        self.assertEqual(
            config["layout_precheck"]["staging_final_confirmation_frames"], 1
        )
        self.assertEqual(
            config["protocol_version"],
            "ac_selective_smoothing_free_single_crossing_pilot_v3",
        )
        self.assertFalse(config["crossing"]["participant_prescribed"])
        self.assertTrue(
            config["crossing"]["actual_hand_direction_speed_gap_recorded"]
        )
        self.assertFalse(
            config["direct_live_scenario"][
                "participant_visible_crossing_markers"
            ]
        )
        self.assertFalse(config["layout_precheck"]["required"])
        self.assertEqual(
            config["direct_live_scenario"]["semantics"],
            DIRECT_LIVE_LAYOUT_SEMANTICS,
        )
        self.assertEqual(
            config["direct_live_scenario"]["corridor_semantics"],
            DIRECT_LIVE_CORRIDOR_SEMANTICS,
        )
        self.assertEqual(
            config["direct_live_scenario"]["fixed_layout_seed"], 10011
        )
        self.assertEqual(
            config["direct_live_scenario"]["expected_layout_id"],
            "8b06c8502c07d1b3",
        )
        qualification = config["direct_live_scenario"][
            "bc_layout_qualification"
        ]
        self.assertTrue(qualification["strict_success"])
        self.assertTrue(qualification["release_observed"])
        self.assertFalse(qualification["corridor_qualified"])

    def test_direct_live_collection_has_no_runtime_bc_only_precheck_call(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse(
            (
                root / "v3_chan" / "collect_ac_selective_smoothing_feedback.py"
            ).read_text(encoding="utf-8")
        )
        run_collection = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_collection"
        )
        calls = {
            node.func.id
            for node in ast.walk(run_collection)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("_initialize_direct_live_scene", calls)
        self.assertNotIn("_run_bc_only_precheck", calls)

    def test_direct_live_scene_reset_performs_no_robot_step(self) -> None:
        config = load_config()
        direct = config["direct_live_scenario"]
        positions = np.asarray(
            direct["expected_robot_initial_joint_positions_rad"], dtype=float
        )
        velocities = np.zeros_like(positions)
        scene = {
            "cube_names": ("cube_0",),
            "cube_positions_world_m": np.asarray(
                [[0.4747162163257599, 0.1833316683769226, 0.9757500290870667]]
            ),
            "cube_orientations_wxyz": np.asarray([[1.0, 0.0, 0.0, 0.0]]),
            "place_target_position_world_m": np.asarray(
                [0.6000000238418579, -0.25, 0.9757500290870667]
            ),
            "place_target_orientation_wxyz": np.asarray(
                [1.0, 0.0, 0.0, 0.0]
            ),
            "robot_initial_joint_positions_rad": positions,
            "robot_initial_joint_velocities_rad_s": velocities,
        }
        source_configuration = {
            "cube_names": ("cube_0",),
            "cube_positions_world": scene["cube_positions_world_m"],
            "cube_orientations_wxyz": scene["cube_orientations_wxyz"],
            "place_target_position_world": scene[
                "place_target_position_world_m"
            ],
            "place_target_orientation_wxyz": scene[
                "place_target_orientation_wxyz"
            ],
            "robot_initial_joint_positions": positions,
            "robot_initial_joint_velocities": velocities,
        }
        restoration = {"source_configuration": source_configuration}

        class FakeWorld:
            paused = False

            def pause(self) -> None:
                self.paused = True

        class FakeEnv:
            def __init__(self) -> None:
                self.reset_seeds: list[int] = []
                self.world = FakeWorld()
                self.robot = SimpleNamespace(dof_names=tuple(range(9)))

            def reset(self, *, seed: int) -> None:
                self.reset_seeds.append(seed)

            def step(self, *_args, **_kwargs):
                raise AssertionError("direct-live initialization moved the robot")

        env = FakeEnv()
        with patch(
            "v3_chan.collect_ac_selective_smoothing_feedback."
            "_capture_scene_and_restoration",
            return_value=(
                scene,
                restoration,
                direct["expected_layout_id"],
            ),
        ), patch(
            "v3_chan.collect_ac_selective_smoothing_feedback."
            "apply_controlled_hold"
        ) as hold:
            observed_scene, observed_restoration, layout_id = (
                _initialize_direct_live_scene(
                    env=env,
                    scenario={"direct_live_scenario": direct},
                )
            )
        self.assertIs(observed_scene, scene)
        self.assertEqual(layout_id, direct["expected_layout_id"])
        self.assertEqual(env.reset_seeds, [10011])
        self.assertTrue(env.world.paused)
        hold.assert_called_once_with(env.robot, joint_count=9)
        self.assertEqual(
            observed_restoration["source_configuration_sha256"],
            direct["bc_layout_qualification"][
                "source_configuration_sha256"
            ],
        )

    def test_direct_live_corridor_validator_binds_geometry_and_trigger_row(self) -> None:
        class Group(dict):
            def __init__(self, attrs, **datasets):
                super().__init__(datasets)
                self.attrs = attrs

        frame = {
            "semantics": DIRECT_LIVE_CORRIDOR_SEMANTICS,
            "detector_semantics": "free_single_crossing_bilateral_gate_v1",
            "participant_visible": False,
            "participant_prescribed_hand": False,
            "participant_prescribed_direction": False,
            "participant_prescribed_speed": False,
            "participant_prescribed_path": False,
            "center_world_m": [0.4, 0.1, 1.1],
            "tangent_world": [0.0, 1.0, 0.0],
            "endpoint_a_world_m": [0.4, -0.1, 1.1],
            "endpoint_b_world_m": [0.4, 0.3, 1.1],
            "gate_half_span_m": 0.2,
            "gate_radius_m": 0.35,
            "interaction_band_radius_m": 0.22,
        }
        audit = {
            "semantics": DIRECT_LIVE_CORRIDOR_SEMANTICS,
            "phase": "transport",
            "detector_semantics": "free_single_crossing_bilateral_gate_v1",
            "scheduled_crossing_factors_applied": False,
            "controller": "frozen_bc_plus_assigned_cbf",
            "cbf_enabled": True,
            "trigger_event": 5,
            "trigger_observation_source_step": 4,
            "trigger_observation_side": "after",
            "calibration_control_step": 5,
            "trigger_simulation_time_s": 1.0,
            "cue_lead_clock": "monotonic_wall",
            "trigger_ee_position_world_m": [0.4, 0.1, 1.1],
            "geometry_valid": True,
            "internal_detection_frame": frame,
        }
        audit["sha256"] = canonical_sha256({"live_corridor": audit})
        planned = {
            **frame,
            "reference_semantics": DIRECT_LIVE_CORRIDOR_SEMANTICS,
            "live_trigger_corridor": audit,
        }
        attrs = {
            "task_phase": "transport",
            "severity": "shallow",
            "crossing_hand": "left",
            "direction": "right_to_left",
            "speed": "slow",
            "planned_gap_min_m": 0.08,
            "planned_gap_max_m": 0.12,
            "planned_target_gap_m": 0.1,
            "planned_corridor_json": json.dumps(planned),
        }
        group = Group(
            attrs,
            **{
                "control/source_step": np.asarray([4], dtype=np.int64),
                "control/result_step": np.asarray([5], dtype=np.int64),
                "control/sim_time": np.asarray([1.0]),
                "control/trial_state": np.asarray(
                    [TrialState.WAIT_FOR_TARGET_TASK_PHASE.value]
                ),
                "robot/ee_position_world_m": np.asarray(
                    [[0.4, 0.1, 1.1]]
                ),
                "task/controller_event_before": np.asarray(
                    [4], dtype=np.int64
                ),
                "task/controller_event_after": np.asarray(
                    [5], dtype=np.int64
                ),
            },
        )
        validator = _Validator(
            config_path=DEFAULT_CONFIG,
            collection_path=None,
            participant_id=None,
            session_id=None,
            seed=None,
            mode=None,
            allow_partial_path=False,
        )
        validator.config = load_config()
        validator._validate_direct_live_corridor(group, "/trials/trial_000000")
        self.assertEqual(validator.issues, [])

        bad = deepcopy(planned)
        bad["live_trigger_corridor"]["trigger_observation_side"] = "before"
        group.attrs["planned_corridor_json"] = json.dumps(bad)
        validator._validate_direct_live_corridor(group, "/trials/trial_000000")
        self.assertIn(
            "corridor.trigger_binding", {issue.code for issue in validator.issues}
        )

        validator.issues.clear()
        malformed = deepcopy(planned)
        malformed_audit = malformed["live_trigger_corridor"]
        malformed_audit.update(
            {
                "trigger_simulation_time_s": 999.0,
                "trigger_ee_position_world_m": [99.0, 99.0, 99.0],
                "geometry_valid": False,
            }
        )
        malformed_audit.pop("sha256")
        malformed_audit["sha256"] = canonical_sha256(
            {"live_corridor": malformed_audit}
        )
        group.attrs["planned_corridor_json"] = json.dumps(malformed)
        validator._validate_direct_live_corridor(
            group, "/trials/trial_000000"
        )
        codes = {issue.code for issue in validator.issues}
        self.assertIn("corridor.trigger_geometry", codes)
        self.assertIn("corridor.trigger_binding", codes)

    def test_haptics_condition_and_shared_safety_drift_are_rejected(self) -> None:
        for path, value, message in (
            (("feedback", "haptics_enabled"), True, "haptics_enabled"),
            (("conditions", "C_smooth", "lambda_s"), 2.0, "lambda_s"),
            (("shared_cbf", "safe_gap_m"), 0.04, "safe_gap_m"),
        ):
            with self.subTest(path=path):
                config = deepcopy(load_config())
                cursor = config
                for key in path[:-1]:
                    cursor = cursor[key]
                cursor[path[-1]] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_config(config)

    def test_controller_mapping_key_and_value_drift_is_rejected(self) -> None:
        mutations = (
            (
                "missing realtime marker",
                lambda feedback: feedback["realtime_markers"]["crossing_left"].pop(
                    "realtime_safety_concern"
                ),
                "realtime_markers",
            ),
            (
                "swapped realtime marker",
                lambda feedback: feedback["realtime_markers"]["crossing_right"].update(
                    realtime_safety_concern="left.y"
                ),
                "realtime_markers",
            ),
            (
                "missing back chord",
                lambda feedback: feedback["questionnaire_navigation"].pop(
                    "back_chord"
                ),
                "questionnaire_navigation",
            ),
            (
                "changed navigation button",
                lambda feedback: feedback["questionnaire_navigation"].update(
                    previous="left.y"
                ),
                "questionnaire_navigation",
            ),
            (
                "changed emergency chord",
                lambda feedback: feedback["questionnaire_navigation"].update(
                    emergency_abort_chord=["right.b", "right.a"]
                ),
                "questionnaire_navigation",
            ),
            (
                "changed emergency hold",
                lambda feedback: feedback["questionnaire_navigation"].update(
                    emergency_abort_hold_s=1.5
                ),
                "questionnaire_navigation",
            ),
            (
                "extra navigation key",
                lambda feedback: feedback["questionnaire_navigation"].update(
                    fallback="right.a"
                ),
                "questionnaire_navigation",
            ),
        )
        for name, mutate, expected_path in mutations:
            with self.subTest(name=name):
                config = deepcopy(load_config())
                mutate(config["feedback"])
                with self.assertRaisesRegex(ValueError, expected_path):
                    validate_config(config)

    def test_static_semantics_rejects_mapping_and_emergency_drift(self) -> None:
        for field, mutate, expected_code in (
            (
                "realtime_markers",
                lambda feedback: feedback["realtime_markers"]["crossing_left"].update(
                    realtime_safety_concern="right.a"
                ),
                "config.marker_mapping",
            ),
            (
                "questionnaire_navigation",
                lambda feedback: feedback["questionnaire_navigation"].update(
                    emergency_abort_chord=["left.x", "left.y"]
                ),
                "config.questionnaire_navigation",
            ),
        ):
            with self.subTest(field=field):
                config = deepcopy(load_config())
                mutate(config["feedback"])
                validator = _Validator(
                    config_path=DEFAULT_CONFIG,
                    collection_path=None,
                    participant_id=None,
                    session_id=None,
                    seed=None,
                    mode=None,
                    allow_partial_path=False,
                )
                validator.config = config
                validator._validate_static_config_semantics()
                self.assertIn(expected_code, {issue.code for issue in validator.issues})

    def test_decision_context_rejects_missing_or_future_outcomes(self) -> None:
        context = {
            "task_phase": "transport",
            "grasp_state": "grasped",
            "attachment_state": "attached",
            "robot_joint_state": {},
            "ee_pose_velocity": {},
            "cube_pose": {},
            "goal_pose": {},
            "hand_pose_velocity": {},
            "surface_gap_m": 0.08,
            "ttc_s": 0.5,
            "closing_speed_m_s": 0.1,
            "active_candidate": {},
            "nominal_rmpflow_joint_command": [0.0] * 7,
            "bc_raw_action": [0.0] * 5,
            "condition_id": "A_reactive",
            "lambda_s": 0.0,
        }
        validate_decision_context(context)
        with self.assertRaisesRegex(ValueError, "future outcomes"):
            validate_decision_context({**context, "task_success": True})
        missing = dict(context)
        missing.pop("bc_raw_action")
        with self.assertRaisesRegex(ValueError, "bc_raw_action"):
            validate_decision_context(missing)


class ControllerFeedbackTests(unittest.TestCase):
    def test_transition_buffer_writes_one_synchronized_batch(self) -> None:
        class FakeDataset:
            def __init__(self, tail_shape=()) -> None:
                self.shape = (0, *tail_shape)
                self.writes = []

            def resize(self, shape) -> None:
                self.shape = tuple(shape)

            def __setitem__(self, key, value) -> None:
                self.writes.append((key, value))

        class FakeFile:
            flush_count = 0

            def flush(self) -> None:
                self.flush_count += 1

        recorder = object.__new__(OnlineExplicitFeedbackRecorder)
        recorder._row_specs = {
            "scalar": DatasetSpec(np.float64),
            "vector": DatasetSpec(np.int8, (2,)),
            "text": DatasetSpec("str"),
        }
        recorder._datasets = {
            "scalar": FakeDataset(),
            "vector": FakeDataset((2,)),
            "text": FakeDataset(),
        }
        recorder._pending_transitions = [
            {
                "scalar": np.asarray(1.0),
                "vector": np.asarray([1, 0], dtype=np.int8),
                "text": "first",
            },
            {
                "scalar": np.asarray(2.0),
                "vector": np.asarray([0, 1], dtype=np.int8),
                "text": "second",
            },
        ]
        recorder._written_trial_length = 0
        recorder._trial_length = 2
        recorder._active_trial = SimpleNamespace(attrs={})
        recorder._file = FakeFile()

        recorder._flush_transition_buffer()

        self.assertEqual(recorder._pending_transitions, [])
        self.assertEqual(recorder._written_trial_length, 2)
        self.assertEqual(recorder._active_trial.attrs["step_count"], 2)
        self.assertEqual(recorder._file.flush_count, 1)
        self.assertEqual(recorder._datasets["scalar"].shape, (2,))
        self.assertEqual(recorder._datasets["vector"].shape, (2, 2))
        np.testing.assert_array_equal(
            recorder._datasets["scalar"].writes[0][1],
            np.asarray([1.0, 2.0]),
        )
        np.testing.assert_array_equal(
            recorder._datasets["vector"].writes[0][1],
            np.asarray([[1, 0], [0, 1]], dtype=np.int8),
        )
        self.assertEqual(
            recorder._datasets["text"].writes[0][1],
            ["first", "second"],
        )

    def test_direct_live_recorder_rejects_layout_precheck_rows(self) -> None:
        recorder = object.__new__(OnlineExplicitFeedbackRecorder)
        recorder._closed = False
        recorder._direct_live = True
        with self.assertRaisesRegex(RuntimeError, "forbids.*BC-only prechecks"):
            recorder.append_layout_precheck({})

    def test_recorder_accepts_right_ab_markers_for_right_hand_crossing(self) -> None:
        class FakeFile(dict):
            def __init__(self) -> None:
                super().__init__(realtime_markers=object())
                self.attrs: dict[str, int] = {}

            def flush(self) -> None:
                pass

        recorder = object.__new__(OnlineExplicitFeedbackRecorder)
        recorder._closed = False
        recorder._active_trial = object()
        recorder._active_plan = SimpleNamespace(
            trial_id="trial-right",
            encounter_id="encounter-right",
            condition_id="A_reactive",
            lambda_s=0.0,
            crossing_hand="right",
        )
        recorder._participant_id = "participant"
        recorder._session_id = "session"
        recorder._file = FakeFile()
        recorder._marker_ids = set()
        recorder._marker_count = 0
        recorder._trial_marker_counts = {
            REALTIME_SAFETY_CONCERN: 0,
            REALTIME_BEHAVIOR_ANOMALY: 0,
        }

        def marker(
            marker_id: str,
            marker_type: str,
            *,
            controller_hand: str,
            button: str,
        ) -> RealtimeMarkerRecord:
            return RealtimeMarkerRecord(
                marker_id=marker_id,
                trial_id="trial-right",
                participant_id="participant",
                session_id="session",
                encounter_id="encounter-right",
                condition_id="A_reactive",
                lambda_s=0.0,
                marker_type=marker_type,
                simulation_time_s=1.0,
                monotonic_ns=1_000_000_000,
                unix_ns=2_000_000_000,
                control_step=10,
                response_phase="CBF_ACTIVE",
                time_since_response_onset_s=0.1,
                time_since_recovery_onset_s=-1.0,
                controller_hand=controller_hand,
                button=button,
                button_identifier=f"{controller_hand}.{button}",
            )

        with patch(
            "v3_chan.ac_feedback.online_recorder._append_table_row"
        ) as append:
            recorder.append_marker(
                marker(
                    "anomaly",
                    REALTIME_BEHAVIOR_ANOMALY,
                    controller_hand="right",
                    button="a",
                )
            )
            recorder.append_marker(
                marker(
                    "safety",
                    REALTIME_SAFETY_CONCERN,
                    controller_hand="right",
                    button="b",
                )
            )
            with self.assertRaisesRegex(
                RuntimeError, "realtime marker provenance"
            ):
                recorder.append_marker(
                    marker(
                        "wrong-left",
                        REALTIME_BEHAVIOR_ANOMALY,
                        controller_hand="left",
                        button="x",
                    )
                )

        self.assertEqual(append.call_count, 2)
        self.assertEqual(recorder._marker_count, 2)
        self.assertEqual(
            recorder._trial_marker_counts,
            {
                REALTIME_SAFETY_CONCERN: 1,
                REALTIME_BEHAVIOR_ANOMALY: 1,
            },
        )

    def test_physically_validated_stage_visual_is_an_allowed_hand_source(self) -> None:
        self.assertIn("xr_stage_visual", DEFAULT_ALLOWED_SOURCES["left"])
        self.assertIn("xr_stage_visual", DEFAULT_ALLOWED_SOURCES["right"])

    def test_xr_prestartup_settings_precede_profile_extension_start(self) -> None:
        args = xr_prestartup_args(xr_mode="vr", xr_backend="OpenXR")
        self.assertIn("--ext-folder", args)
        self.assertIn("--enable", args)
        self.assertIn("omni.sert.xr.feedback_input", args)
        self.assertIn("--/xr/persistence/enabled=false", args)
        self.assertIn("--/xr/profile/vr/anchorMode=active camera", args)
        self.assertIn("--/xr/profile/vr/adjustForUserHeight=false", args)
        self.assertIn("--/xr/profile/vr/tools/layout=vr", args)
        self.assertNotIn("--/xr/profile/vr/tools/layout=", args)
        self.assertTrue(
            any(value.startswith("--/app/font/customFontPath=") for value in args)
        )
        self.assertTrue(
            any(
                value.startswith("--/app/font/customRegionFiles=[")
                for value in args
            )
        )

    def test_feedback_xr_manifests_keep_vr_layout_without_stock_tools(self) -> None:
        root = Path(__file__).resolve().parents[1]
        extension = (
            root
            / "v3_chan"
            / "ac_feedback"
            / "xr_exts"
            / "omni.sert.xr.feedback_input"
        )
        config = (extension / "config" / "extension.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'xr.manifests."omni.sert.xr.feedback_input" = "xrmanifests"',
            config,
        )
        manifests = sorted((extension / "xrmanifests" / "action_maps").glob("*.json"))
        self.assertEqual(len(manifests), 2)
        dominant_hands = set()
        for path in manifests:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 1)
            self.assertGreater(payload["priority"], 30)
            self.assertEqual(
                payload["name"],
                f"SeRT Feedback Oculus ({payload['dominantHand'].title()})",
            )
            self.assertEqual(payload["toolsLayout"], "vr")
            self.assertEqual(payload["tools"], [])
            self.assertEqual(payload["actions"], {})
            dominant_hands.add(payload["dominantHand"])
            self.assertEqual(
                payload["tags"],
                [
                    {
                        "inputDevice": "/user/hand/left",
                        "tag": "left_oculus_touch_components",
                    },
                    {
                        "inputDevice": "/user/hand/right",
                        "tag": "right_oculus_touch_components",
                    },
                ],
            )
        self.assertEqual(dominant_hands, {"left", "right"})

    def test_feedback_action_map_runtime_gate_accepts_only_tool_free_map(self) -> None:
        class ActionMap:
            @staticmethod
            def get_name():
                return "SeRT Feedback Oculus (Right)"

            @staticmethod
            def get_tool_layout():
                return "vr"

            @staticmethod
            def get_dominant_hand():
                return "right"

            @staticmethod
            def get_tool_list():
                return ()

            @staticmethod
            def get_action_map():
                return ()

        class Core:
            @staticmethod
            def get_action_map():
                return ActionMap()

        self.assertEqual(
            verify_feedback_only_action_map(Core()),
            {
                "name": "SeRT Feedback Oculus (Right)",
                "tools_layout": "vr",
                "dominant_hand": "right",
                "tool_count": 0,
                "action_count": 0,
            },
        )

    def test_xr_token_text_accepts_isaac_wrapper_representation(self) -> None:
        class Token:
            @staticmethod
            def __str__():
                return 'XRToken(name: "vr")'

        self.assertEqual(_xr_token_text(Token()), "vr")

    def test_feedback_action_map_runtime_gate_rejects_stock_map(self) -> None:
        class StockMap:
            @staticmethod
            def get_name():
                return "VR Oculus (Right)"

            @staticmethod
            def get_tool_layout():
                return "vr"

            @staticmethod
            def get_dominant_hand():
                return "right"

            @staticmethod
            def get_tool_list():
                return ("select", "menu")

            @staticmethod
            def get_action_map():
                return (object(),)

        class Core:
            @staticmethod
            def get_action_map():
                return StockMap()

        with self.assertRaisesRegex(
            RuntimeError, "feedback-only XR action-map isolation failed"
        ):
            verify_feedback_only_action_map(Core())

    def test_feedback_action_map_runtime_gate_rejects_spoofed_prefix(self) -> None:
        class SpoofedMap:
            @staticmethod
            def get_name():
                return "SeRT Feedback Oculus (Right) unreviewed"

            @staticmethod
            def get_tool_layout():
                return "vr"

            @staticmethod
            def get_dominant_hand():
                return "right"

            @staticmethod
            def get_tool_list():
                return ()

            @staticmethod
            def get_action_map():
                return ()

        class Core:
            @staticmethod
            def get_action_map():
                return SpoofedMap()

        with self.assertRaisesRegex(RuntimeError, "selected map"):
            verify_feedback_only_action_map(Core())

    def test_live_vr_action_map_guard_is_rechecked_on_every_sample(self) -> None:
        class Avatar:
            @staticmethod
            def update_with_status():
                return ("head", "left", "right")

            @staticmethod
            def capture_initial_hmd_pos():
                return None

        class Watchdog:
            @staticmethod
            def evaluate(samples):
                return samples

        provider = LiveVRTrackingProvider(Watchdog())
        provider.avatar = Avatar()
        with patch(
            "v3_chan.ac_feedback.live_vr.verify_feedback_only_action_map",
            return_value={"name": "SeRT Feedback Oculus (Right)"},
        ) as verify:
            self.assertEqual(
                provider.arm_feedback_action_map_guard(),
                {"name": "SeRT Feedback Oculus (Right)"},
            )
            self.assertEqual(provider.sample(), ("head", "left", "right"))
        self.assertEqual(verify.call_count, 2)
        verify.assert_any_call()
        verify.assert_any_call(emit_success=False)

    def test_validator_requires_tool_free_action_map_audit(self) -> None:
        validator = _Validator(
            config_path=DEFAULT_CONFIG,
            collection_path=None,
            participant_id=None,
            session_id=None,
            seed=None,
            mode=None,
            allow_partial_path=False,
        )
        valid_audit = {
            "name": "SeRT Feedback Oculus (Left)",
            "tools_layout": "vr",
            "dominant_hand": "left",
            "tool_count": 0,
            "action_count": 0,
        }
        validator._validate_xr_action_map(
            {"xr_feedback_action_map_json": json.dumps(valid_audit)}
        )
        self.assertEqual(validator.issues, [])
        invalid_audit = dict(valid_audit, action_count=1)
        validator._validate_xr_action_map(
            {"xr_feedback_action_map_json": json.dumps(invalid_audit)}
        )
        self.assertIn("xr.action_map", {issue.code for issue in validator.issues})

    def test_korean_font_region_covers_participant_visible_literals(self) -> None:
        root = Path(__file__).resolve().parents[1]
        region_path = (
            root / "v3_chan" / "ac_feedback" / "assets" / "korean_ui_chars.txt"
        )
        region = set(region_path.read_text(encoding="utf-8"))
        required = set()
        for relative in (
            "v3_chan/ac_feedback/xr_feedback.py",
            "v3_chan/collect_ac_selective_smoothing_feedback.py",
        ):
            tree = ast.parse((root / relative).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    required.update(
                        char
                        for char in node.value
                        if "\uac00" <= char <= "\ud7a3"
                    )
        self.assertEqual(required - region, set())

    def test_runtime_xr_settings_use_isolated_profile_paths(self) -> None:
        class Settings:
            def __init__(self) -> None:
                self.values = {}

            def set(self, key, value) -> None:
                self.values[key] = value

        settings = Settings()
        result = _configure_xr_profile_settings(
            settings, profile_name="vr", xr_backend="OpenXR"
        )
        self.assertIsNone(result)
        for root in ("/defaults/xr/profile/vr", "/xr/profile/vr"):
            self.assertEqual(settings.values[f"{root}/anchorMode"], "active camera")
            self.assertIs(settings.values[f"{root}/adjustForUserHeight"], False)
            self.assertEqual(settings.values[f"{root}/tools/layout"], "vr")

    def test_live_vr_prefers_proven_camera_placement_over_anchor_edit(self) -> None:
        class Avatar:
            notified = False

            @staticmethod
            def update_with_status():
                return ("head", "left", "right")

            @staticmethod
            def capture_initial_hmd_pos():
                return np.array([0.0, 1.7, 0.0])

            def notify_anchor_applied(self):
                self.notified = True

        class Watchdog:
            @staticmethod
            def evaluate(samples):
                return samples

        provider = LiveVRTrackingProvider(Watchdog())
        provider.avatar = Avatar()
        provider._simulation_app = object()
        provider._view_target_world = np.array([0.4, 0.0, 0.75])
        with patch(
            "v3_chan.ac_feedback.live_vr.schedule_live_vr_camera_view",
            return_value=True,
        ) as camera, patch(
            "v3_chan.ac_feedback.live_vr.set_live_vr_anchor"
        ) as anchor:
            self.assertEqual(provider.sample(), ("head", "left", "right"))
        camera.assert_called_once()
        anchor.assert_not_called()
        self.assertTrue(provider.anchor_applied)
        self.assertTrue(provider.avatar.notified)
        self.assertEqual(provider.placement_method, "xr_camera_teleport")

    def test_hmd_startup_gate_does_not_wait_for_controller_poses(self) -> None:
        class Snapshot:
            valid_mask = np.array([1, 0, 0], dtype=np.int8)
            invalid_reasons = (
                "left:pose_invalid",
                "right:pose_invalid",
            )

        class App:
            updates = 0

            @staticmethod
            def is_running():
                return True

            def update(self):
                self.updates += 1

        provider = LiveVRTrackingProvider(object())
        provider._anchor_applied = True
        provider.refresh_observation_state = lambda: Snapshot()
        app = App()
        with patch("v3_chan.ac_feedback.live_vr.time.sleep"):
            result = provider.wait_until_hmd_ready(
                app,
                timeout_s=1.0,
                stable_frames=2,
            )
        self.assertIsInstance(result, Snapshot)
        self.assertEqual(app.updates, 2)

    def test_cached_pose_age_classification_excludes_xr_source_staleness(self) -> None:
        self.assertTrue(
            _cached_pose_age_only(
                TrackingDropout(
                    ("head:pose_stale", "left:effective_pose_stale"), None
                )
            )
        )
        self.assertFalse(
            _cached_pose_age_only(
                TrackingDropout(("head:source_pose_stale",), None)
            )
        )

    def test_post_step_software_age_does_not_abort_valid_acquired_sample(self) -> None:
        snapshot = object()
        test_case = self

        class Watchdog:
            current_checks = 0
            valid_checks = 0

            def require_current(self, value, **kwargs):
                test_case.assertIs(value, snapshot)
                test_case.assertEqual(kwargs["now_monotonic_ns"], 123)
                self.current_checks += 1
                raise TrackingDropout(
                    (
                        "head:pose_stale",
                        "left:effective_pose_stale",
                        "right:pose_stale",
                    ),
                    None,
                )

            def require_valid(self, value):
                test_case.assertIs(value, snapshot)
                self.valid_checks += 1

        watchdog = Watchdog()
        _require_post_step_tracking(
            watchdog=watchdog,
            tracking=snapshot,
            post_step_monotonic_ns=123,
        )
        self.assertEqual(watchdog.current_checks, 1)
        self.assertEqual(watchdog.valid_checks, 1)

    def test_post_step_real_tracking_failure_remains_fail_closed(self) -> None:
        class Watchdog:
            valid_checks = 0

            @staticmethod
            def require_current(_value, **_kwargs):
                raise TrackingDropout(("left:pose_invalid",), None)

            def require_valid(self, _value):
                self.valid_checks += 1

        watchdog = Watchdog()
        with self.assertRaisesRegex(TrackingDropout, "left:pose_invalid"):
            _require_post_step_tracking(
                watchdog=watchdog,
                tracking=object(),
                post_step_monotonic_ns=123,
            )
        self.assertEqual(watchdog.valid_checks, 0)

    def test_expired_cached_observation_is_rebuilt_before_control(self) -> None:
        old_snapshot = object()
        new_snapshot = object()

        class Watchdog:
            @staticmethod
            def require_current(value):
                if value is old_snapshot:
                    raise TrackingDropout(("head:pose_stale",), None)
                self.assertIs(value, new_snapshot)

        class Robot:
            dof_names = [f"joint_{index}" for index in range(9)]

        class Env:
            robot = Robot()

            @staticmethod
            def refresh_observation():
                return np.ones(4, dtype=np.float32), {"step": 7}

        class Provider:
            observation_snapshot = new_snapshot

        with patch(
            "v3_chan.collect_ac_selective_smoothing_feedback.apply_controlled_hold"
        ) as hold:
            obs, info, tracking = _refresh_expired_control_observation(
                env=Env(),
                provider=Provider(),
                watchdog=Watchdog(),
                obs=np.zeros(4, dtype=np.float32),
                info={"step": 6},
                tracking=old_snapshot,
            )
        hold.assert_called_once()
        np.testing.assert_array_equal(obs, np.ones(4, dtype=np.float32))
        self.assertEqual(info["step"], 7)
        self.assertIs(tracking, new_snapshot)

    def test_observation_aged_by_preflight_is_rebuilt_once(self) -> None:
        old_snapshot = object()
        preflight_snapshot = object()
        refreshed_snapshot = object()

        class Watchdog:
            calls: list[object] = []

            def require_current(self, value, **_kwargs):
                self.calls.append(value)
                if value is old_snapshot:
                    raise TrackingDropout(
                        (
                            "head:pose_stale",
                            "head:effective_pose_stale",
                            "left:pose_stale",
                            "left:effective_pose_stale",
                            "right:pose_stale",
                            "right:effective_pose_stale",
                        ),
                        None,
                    )

        class Robot:
            dof_names = [f"joint_{index}" for index in range(9)]

        class Provider:
            observation_snapshot = old_snapshot

            @staticmethod
            def sample():
                return preflight_snapshot

        provider = Provider()

        class Env:
            robot = Robot()
            refresh_count = 0

            def refresh_observation(self):
                self.refresh_count += 1
                provider.observation_snapshot = refreshed_snapshot
                return np.ones(4, dtype=np.float32), {"step": 8}

        env = Env()
        watchdog = Watchdog()
        with patch(
            "v3_chan.collect_ac_selective_smoothing_feedback."
            "apply_controlled_hold"
        ) as hold:
            obs, info, tracking, preflight_ns = (
                _prepare_current_control_observation(
                    env=env,
                    provider=provider,
                    watchdog=watchdog,
                    obs=np.zeros(4, dtype=np.float32),
                    info={"step": 7},
                    tracking=old_snapshot,
                )
            )

        hold.assert_called_once_with(env.robot, joint_count=9)
        self.assertEqual(env.refresh_count, 1)
        np.testing.assert_array_equal(obs, np.ones(4, dtype=np.float32))
        self.assertEqual(info["step"], 8)
        self.assertIs(tracking, refreshed_snapshot)
        self.assertGreater(preflight_ns, 0)
        self.assertIs(watchdog.calls[0], preflight_snapshot)
        self.assertIn(old_snapshot, watchdog.calls)
        self.assertIs(watchdog.calls[-1], refreshed_snapshot)

    def test_observation_without_apply_reserve_is_rebuilt_while_current(
        self,
    ) -> None:
        old_snapshot = object()
        preflight_snapshot = object()
        refreshed_snapshot = object()

        class Watchdog:
            max_pose_age_ms = 100.0

            def __init__(self) -> None:
                self.old_check_times: list[int | None] = []

            def require_current(self, value, **kwargs):
                if value is old_snapshot:
                    checked_at = kwargs.get("now_monotonic_ns")
                    self.old_check_times.append(checked_at)
                    if checked_at is not None and checked_at > 120_000_000:
                        raise TrackingDropout(
                            ("head:effective_pose_stale",), None
                        )

        class Robot:
            dof_names = [f"joint_{index}" for index in range(9)]

        class Provider:
            observation_snapshot = old_snapshot

            @staticmethod
            def sample():
                return preflight_snapshot

        provider = Provider()

        class Env:
            robot = Robot()

            @staticmethod
            def refresh_observation():
                provider.observation_snapshot = refreshed_snapshot
                return np.ones(4, dtype=np.float32), {"step": 8}

        watchdog = Watchdog()
        with patch(
            "v3_chan.collect_ac_selective_smoothing_feedback."
            "time.monotonic_ns",
            side_effect=[100_000_000, 100_000_000, 100_000_000],
        ), patch(
            "v3_chan.collect_ac_selective_smoothing_feedback."
            "apply_controlled_hold"
        ) as hold:
            obs, info, tracking, preflight_ns = (
                _prepare_current_control_observation(
                    env=Env(),
                    provider=provider,
                    watchdog=watchdog,
                    obs=np.zeros(4, dtype=np.float32),
                    info={"step": 7},
                    tracking=old_snapshot,
                )
            )

        hold.assert_called_once()
        self.assertEqual(watchdog.old_check_times, [130_000_000])
        np.testing.assert_array_equal(obs, np.ones(4, dtype=np.float32))
        self.assertEqual(info["step"], 8)
        self.assertIs(tracking, refreshed_snapshot)
        self.assertEqual(preflight_ns, 100_000_000)

    def test_real_tracking_failure_after_preflight_is_not_retried(self) -> None:
        invalid_snapshot = object()
        preflight_snapshot = object()

        class Watchdog:
            @staticmethod
            def require_current(value, **_kwargs):
                if value is invalid_snapshot:
                    raise TrackingDropout(("left:pose_invalid",), None)

        class Provider:
            observation_snapshot = invalid_snapshot

            @staticmethod
            def sample():
                return preflight_snapshot

        class Robot:
            dof_names = [f"joint_{index}" for index in range(9)]

        class Env:
            robot = Robot()
            refresh_count = 0

            def refresh_observation(self):
                self.refresh_count += 1
                raise AssertionError("real tracking failures must fail closed")

        env = Env()
        with patch(
            "v3_chan.collect_ac_selective_smoothing_feedback."
            "apply_controlled_hold"
        ) as hold, self.assertRaises(TrackingDropout):
            _prepare_current_control_observation(
                env=env,
                provider=Provider(),
                watchdog=Watchdog(),
                obs=np.zeros(4, dtype=np.float32),
                info={"step": 7},
                tracking=invalid_snapshot,
            )
        self.assertEqual(env.refresh_count, 0)
        hold.assert_not_called()

    def test_preapply_guard_rechecks_stop_after_blocking_xr_sample(self) -> None:
        class Action:
            def __init__(self, *, joint_positions, joint_velocities) -> None:
                self.joint_positions = np.asarray(
                    joint_positions, dtype=np.float64
                )
                self.joint_velocities = np.asarray(
                    joint_velocities, dtype=np.float64
                )

        class Robot:
            dof_names = tuple(f"joint_{index}" for index in range(9))

            def __init__(self) -> None:
                self.calls = []
                self.applied = Action(
                    joint_positions=np.zeros(9),
                    joint_velocities=np.zeros(9),
                )

            @staticmethod
            def get_joint_positions():
                return np.arange(9, dtype=np.float64)

            def apply_action(self, action) -> None:
                self.calls.append(action)
                self.applied = action

            def get_applied_action(self):
                return self.applied

        stop = {"reason": ""}

        class Provider:
            observation_snapshot = object()

            @staticmethod
            def sample():
                stop["reason"] = "signal_15"
                return object()

        class Watchdog:
            @staticmethod
            def require_current(_snapshot):
                return None

        robot = Robot()
        trace = SimpleNamespace(
            robot=robot, installed=True, step_active=True
        )
        guard = PreApplyTrackingGuard(
            robot=robot,
            provider=Provider(),
            watchdog=Watchdog(),
            trace=trace,
            abort_reason_fn=lambda: stop["reason"],
        ).install()
        candidate = Action(
            joint_positions=np.full(9, 99.0),
            joint_velocities=np.full(9, 99.0),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "signal_15"):
                robot.apply_action(candidate)
        finally:
            guard.uninstall()

        self.assertEqual(len(robot.calls), 1)
        self.assertIsNot(robot.calls[0], candidate)
        np.testing.assert_array_equal(
            robot.calls[0].joint_positions, np.arange(9, dtype=np.float64)
        )
        np.testing.assert_array_equal(
            robot.calls[0].joint_velocities, np.zeros(9, dtype=np.float64)
        )

    def test_tracking_clear_wait_does_not_age_final_sample_with_extra_frame(self) -> None:
        snapshot = object()

        class World:
            def pause(self):
                pass

            def play(self):
                pass

        class Env:
            world = World()
            robot = object()

            @staticmethod
            def refresh_observation():
                return np.zeros(4, dtype=np.float32), {
                    "geometry_valid": True,
                    "left_end_effector_surface_gap_m": 1.0,
                    "right_end_effector_surface_gap_m": 1.0,
                }

        class Watchdog:
            @staticmethod
            def require_current(value):
                self.assertIs(value, snapshot)

        class Provider:
            observation_snapshot = snapshot
            watchdog = Watchdog()

        class App:
            updates = 0

            def update(self):
                self.updates += 1

        app = App()
        statuses: list[str] = []
        with patch(
            "v3_chan.collect_ac_selective_smoothing_feedback.apply_controlled_hold"
        ):
            _obs, _info, returned = _wait_for_tracking_and_clear(
                env=Env(),
                provider=Provider(),
                simulation_app=app,
                joint_count=9,
                timeout_s=1.0,
                staging_dwell_s=0.0,
                final_confirmation_frames=1,
                clear_gap_m=0.13,
                proximal_monitor=None,
                stop_requested=lambda: None,
                status_callback=statuses.append,
                status_label="준비 검사 1/12",
            )
        self.assertIs(returned, snapshot)
        self.assertEqual(app.updates, 0)
        self.assertIn("준비 검사 1/12", statuses[-1])
        self.assertIn("바로 시작합니다", statuses[-1])
        self.assertNotIn("연속 확인", statuses[-1])

    def test_tracking_clear_wait_uses_wall_dwell_without_dropout_reset(self) -> None:
        snapshots = iter(("bad-0", "good-2", "bad-3", "good-5"))

        class World:
            paused = 0
            played = 0

            def pause(self):
                self.paused += 1

            def play(self):
                self.played += 1

        class Provider:
            observation_snapshot = None

            class Watchdog:
                @staticmethod
                def require_current(value):
                    if str(value).startswith("bad"):
                        raise TrackingDropout(("left:pose_invalid",), value)

            watchdog = Watchdog()

        provider = Provider()

        class Env:
            world = World()
            robot = object()

            @staticmethod
            def refresh_observation():
                provider.observation_snapshot = next(snapshots)
                return np.zeros(4, dtype=np.float32), {
                    "geometry_valid": True,
                    "left_end_effector_surface_gap_m": 1.0,
                    "right_end_effector_surface_gap_m": 1.0,
                }

        class App:
            updates = 0

            def update(self):
                self.updates += 1

        app = App()
        statuses: list[str] = []
        with (
            patch(
                "v3_chan.collect_ac_selective_smoothing_feedback."
                "apply_controlled_hold"
            ),
            patch(
                "v3_chan.collect_ac_selective_smoothing_feedback.time.monotonic",
                side_effect=(0.0, 0.0, 2.0, 3.0, 5.0),
            ),
            patch("v3_chan.collect_ac_selective_smoothing_feedback.time.sleep"),
        ):
            _obs, _info, returned = _wait_for_tracking_and_clear(
                env=Env(),
                provider=provider,
                simulation_app=app,
                joint_count=9,
                timeout_s=10.0,
                staging_dwell_s=5.0,
                final_confirmation_frames=1,
                clear_gap_m=0.13,
                proximal_monitor=None,
                stop_requested=lambda: None,
                status_callback=statuses.append,
            )

        self.assertEqual(returned, "good-5")
        self.assertEqual(app.updates, 3)
        self.assertEqual(Env.world.played, 1)
        self.assertTrue(any("머무른 시간 3.0/5.0초" in text for text in statuses))
        self.assertIn("바로 시작합니다", statuses[-1])
        self.assertTrue(all("연속 확인" not in text for text in statuses))

    def test_tracking_clear_wait_does_not_repeat_dwell_after_final_dropout(self) -> None:
        snapshots = iter(("bad-after-dwell", "good-after-dwell"))

        class World:
            def pause(self):
                pass

            def play(self):
                pass

        class Provider:
            observation_snapshot = None

            class Watchdog:
                @staticmethod
                def require_current(value):
                    if value == "bad-after-dwell":
                        raise TrackingDropout(("right:pose_invalid",), value)

            watchdog = Watchdog()

        provider = Provider()

        class Env:
            world = World()
            robot = object()

            @staticmethod
            def refresh_observation():
                provider.observation_snapshot = next(snapshots)
                return np.zeros(4, dtype=np.float32), {
                    "geometry_valid": True,
                    "left_end_effector_surface_gap_m": 1.0,
                    "right_end_effector_surface_gap_m": 1.0,
                }

        class App:
            updates = 0

            def update(self):
                self.updates += 1

        app = App()
        statuses: list[str] = []
        with (
            patch(
                "v3_chan.collect_ac_selective_smoothing_feedback."
                "apply_controlled_hold"
            ),
            patch(
                "v3_chan.collect_ac_selective_smoothing_feedback.time.monotonic",
                side_effect=(0.0, 5.0, 5.1),
            ),
            patch("v3_chan.collect_ac_selective_smoothing_feedback.time.sleep"),
        ):
            _obs, _info, returned = _wait_for_tracking_and_clear(
                env=Env(),
                provider=provider,
                simulation_app=app,
                joint_count=9,
                timeout_s=10.0,
                staging_dwell_s=5.0,
                final_confirmation_frames=1,
                clear_gap_m=0.13,
                proximal_monitor=None,
                stop_requested=lambda: None,
                status_callback=statuses.append,
            )

        self.assertEqual(returned, "good-after-dwell")
        self.assertEqual(app.updates, 1)
        self.assertIn("5초 유지 완료", statuses[0])
        self.assertIn("바로 시작합니다", statuses[-1])

    def test_controller_release_preflight_reacquires_tracking(self) -> None:
        class Watchdog:
            calls = 0

            def require_current(self, _snapshot):
                self.calls += 1
                if self.calls == 1:
                    raise TrackingDropout(("left:pose_invalid",), None)

        class Provider:
            watchdog = Watchdog()
            samples = 0

            def sample(self):
                self.samples += 1
                return object()

        class Source:
            last_error = ""

            @staticmethod
            def read():
                return ControllerButtonSnapshot(
                    left_connected=True,
                    right_connected=True,
                    left_x=False,
                    left_y=False,
                    right_a=False,
                    right_b=False,
                )

        class Feedback:
            instructions: list[str] = []

            @staticmethod
            def set_crossing_hand(_hand):
                pass

            def set_instruction(self, text):
                self.instructions.append(str(text))

            @staticmethod
            def update(_snapshot, _clocks):
                pass

        class World:
            current_time = 0.0

        class Env:
            world = World()
            step_count = 0

        class App:
            @staticmethod
            def update():
                pass

        provider = Provider()
        feedback = Feedback()
        with patch(
            "v3_chan.collect_ac_selective_smoothing_feedback.time.sleep"
        ):
            _wait_for_controller_release(
                source=Source(),
                feedback=feedback,
                provider=provider,
                simulation_app=App(),
                env=Env(),
                timeout_s=1.0,
                stop_requested=lambda: None,
            )
        self.assertEqual(provider.samples, 4)
        self.assertIn("왼손 위치 추적", feedback.instructions[0])
        self.assertIn("완료 3/3", feedback.instructions[-1])

    def test_xr_ui_open_failure_preserves_backend_detail(self) -> None:
        class FailedRenderer:
            available = False
            last_error = "TypeError: Label requires a string"

            @staticmethod
            def open() -> bool:
                return False

        renderer = FailedRenderer()
        controller = XRFeedbackController()
        with patch(
            "v3_chan.ac_feedback.xr_feedback.XRSceneViewTextRenderer",
            return_value=renderer,
        ):
            self.assertFalse(controller.attach_xr_ui())
        self.assertEqual(controller.last_error, renderer.last_error)

    def test_xr_scene_view_renderer_updates_retained_label_text(self) -> None:
        class Label:
            text = ""

        label = Label()
        renderer = XRSceneViewTextRenderer()
        renderer._label = label
        renderer.update("updated feedback")
        self.assertEqual(label.text, "updated feedback")
        self.assertEqual((renderer.width, renderer.height), (900, 700))
        self.assertEqual(renderer.world_position, (0.35, 0.0, 1.50))

    def test_setup_instruction_does_not_append_generic_idle_line(self) -> None:
        controller = XRFeedbackController()
        controller.set_instruction("준비 검사 1/12")
        self.assertEqual(controller.renderer_text, "준비 검사 1/12")

    def test_dismissed_terminal_questionnaire_exposes_resume_instruction(self) -> None:
        controller = XRFeedbackController()
        saved_result = object()
        controller.questionnaire._state = "completed"
        controller.questionnaire._result = saved_result
        controller.set_instruction("로봇이 남은 작업을 완료 중입니다.")
        self.assertEqual(controller.renderer_text, "응답이 기록되었습니다.")

        controller.dismiss_questionnaire()

        self.assertEqual(controller.questionnaire.state, "idle")
        self.assertIs(controller.latest_questionnaire_result, saved_result)
        self.assertEqual(
            controller.renderer_text, "로봇이 남은 작업을 완료 중입니다."
        )

    def test_controller_input_watchdog_distinguishes_health_and_dropout(self) -> None:
        watchdog = ControllerInputWatchdog()
        healthy = ControllerButtonSnapshot(
            left_connected=True,
            right_connected=True,
            left_x=False,
            left_y=True,
            right_a=False,
            right_b=True,
        )
        self.assertIs(watchdog.require_available(healthy), healthy)
        self.assertEqual(watchdog.unavailable_reasons(healthy), ())

        disconnected = ControllerButtonSnapshot(
            left_connected=False,
            right_connected=True,
            right_a=False,
            right_b=False,
        )
        with self.assertRaises(ControllerInputUnavailable) as caught:
            watchdog.require_available(disconnected)
        self.assertEqual(
            caught.exception.reasons, ("left_controller_disconnected",)
        )

        unknown_button = ControllerButtonSnapshot(
            left_connected=True,
            right_connected=True,
            left_x=False,
            left_y=False,
            right_a=None,
            right_b=False,
        )
        with self.assertRaises(ControllerInputUnavailable) as caught:
            watchdog.require_available(unknown_button)
        self.assertEqual(caught.exception.reasons, ("right.a_state_unknown",))

    def test_controller_input_dropout_applies_verified_hold_before_abort(self) -> None:
        class Source:
            last_error = "right A unavailable"

            @staticmethod
            def read() -> ControllerButtonSnapshot:
                return ControllerButtonSnapshot(
                    left_connected=True,
                    right_connected=True,
                    left_x=False,
                    left_y=False,
                    right_a=None,
                    right_b=False,
                )

        robot = object()
        with patch(
            "v3_chan.collect_ac_selective_smoothing_feedback.apply_controlled_hold"
        ) as hold:
            with self.assertRaisesRegex(
                CollectionAbort, "controller_input_unavailable_pre_step"
            ):
                _read_required_controller_inputs(
                    source=Source(),
                    input_watchdog=ControllerInputWatchdog(),
                    robot=robot,
                    joint_count=9,
                    sample_phase="pre_step",
                )
        hold.assert_called_once_with(robot, joint_count=9)

    def test_realtime_mapping_uses_right_controller_independent_of_crossing_hand(self) -> None:
        config = load_config()["feedback"]["realtime_markers"]
        mapping = RealtimeButtonMapping.from_mapping({
            "left": config["crossing_left"],
            "right": config["crossing_right"],
        })
        self.assertEqual(
            tuple(button.key for button in mapping.for_crossing("left")),
            ("right.b", "right.a"),
        )
        self.assertEqual(
            tuple(button.key for button in mapping.for_crossing("right")),
            ("right.b", "right.a"),
        )
        marker_input = RealtimeMarkerInput(
            mapping=mapping, crossing_hand="left", id_factory=lambda: "marker-id",
        )
        records = marker_input.records_for_edges(
            ("right.a", "right.b"), _clock(1)
        )
        self.assertEqual(
            [record.marker_type for record in records],
            [REALTIME_BEHAVIOR_ANOMALY, REALTIME_SAFETY_CONCERN],
        )
        self.assertTrue(all(record.controller_hand == "right" for record in records))

        marker_input.set_crossing_hand("right")
        records = marker_input.records_for_edges(
            ("right.a", "right.b"), _clock(2)
        )
        self.assertEqual(
            [record.marker_type for record in records],
            [REALTIME_BEHAVIOR_ANOMALY, REALTIME_SAFETY_CONCERN],
        )
        self.assertTrue(all(record.controller_hand == "right" for record in records))

    def test_realtime_mapping_rejects_crossing_dependent_controller_bindings(self) -> None:
        config = load_config()["feedback"]["realtime_markers"]
        mapping = {
            "left": dict(config["crossing_left"]),
            "right": dict(config["crossing_right"]),
        }
        mapping["right"][REALTIME_SAFETY_CONCERN] = "left.x"
        mapping["right"][REALTIME_BEHAVIOR_ANOMALY] = "left.y"
        with self.assertRaisesRegex(ValueError, "right controller"):
            RealtimeButtonMapping.from_mapping(mapping)

    def test_realtime_feedback_ui_states_right_a_and_b_meanings(self) -> None:
        controller = XRFeedbackController()
        controller.set_crossing_hand("right")
        self.assertIn("오른손 A: 이상한 궤적", controller.renderer_text)
        self.assertIn("오른손 B: 위험한 궤적", controller.renderer_text)

    def test_realtime_mapping_does_not_fill_missing_bindings(self) -> None:
        config = load_config()["feedback"]["realtime_markers"]
        mapping = {
            "left": dict(config["crossing_left"]),
            "right": dict(config["crossing_right"]),
        }
        mapping["left"].pop(REALTIME_SAFETY_CONCERN)
        with self.assertRaisesRegex(ValueError, "exactly"):
            RealtimeButtonMapping.from_mapping(mapping)

    def test_configured_questionnaire_mapping_and_vocabulary_are_exact(self) -> None:
        feedback = load_config()["feedback"]
        mapping = QuestionnaireButtonMapping.from_mapping(
            feedback["questionnaire_navigation"]
        )
        self.assertEqual(mapping.previous.key, "left.x")
        self.assertEqual(mapping.next.key, "left.y")
        self.assertEqual(mapping.select.key, "right.a")
        self.assertEqual(mapping.submit.key, "right.b")
        self.assertEqual(mapping.back_keys, frozenset(("left.x", "left.y")))
        self.assertEqual(tuple(feedback["q1_responses"]), Q1_RESPONSE_CODES)
        self.assertEqual(tuple(feedback["modification_reasons"]), REJECTION_REASON_IDS)
        self.assertEqual(tuple(feedback["questions"])[1:], tuple(LIKERT_QUESTIONS_KO))

    def test_questionnaire_mapping_does_not_default_back_chord(self) -> None:
        navigation = dict(load_config()["feedback"]["questionnaire_navigation"])
        navigation.pop("back_chord")
        with self.assertRaisesRegex(ValueError, "back_chord"):
            QuestionnaireButtonMapping.from_mapping(navigation)

    def test_needs_modification_requires_at_least_one_reason(self) -> None:
        questionnaire = QuestionnaireFSM(timeout_s=30.0, id_factory=lambda: "query-id")
        questionnaire.start(_clock(0), context_id="trial-id")
        for index in range(6):
            questionnaire.handle_edges(("right.a",), _clock(1 + index * 2))
            questionnaire.handle_edges(("right.b",), _clock(2 + index * 2))
        self.assertEqual(questionnaire.state, "rejection_reasons")
        self.assertIsNone(questionnaire.handle_edges(("right.b",), _clock(13)))
        self.assertTrue(questionnaire.active)
        questionnaire.handle_edges(("right.a",), _clock(14))
        result = questionnaire.handle_edges(("right.b",), _clock(15))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.completion_status, "completed")
        self.assertEqual(result.response_disposition, "answered")
        self.assertEqual(result.q1_response, "needs_modification")
        self.assertEqual(result.rejection_reason_ids, ("too_close_or_late",))
        self.assertEqual(result.first_input_control_step, 1)
        self.assertEqual(len(result.answers), 7)
        self.assertTrue(
            all(answer.answer_status == "confirmed" for answer in result.answers)
        )
        reason_answer = result.answers[-1]
        self.assertEqual(reason_answer.question_id, "rejection_reasons")
        self.assertEqual(reason_answer.answer_status, "confirmed")
        self.assertEqual(reason_answer.prompt_shown_control_step, 12)
        self.assertEqual(reason_answer.first_input_control_step, 13)
        self.assertEqual(reason_answer.confirmed_control_step, 15)
        self.assertEqual(reason_answer.response_latency_ms, 300.0)
        self.assertEqual(reason_answer.input_device, "vr_controller")
        self.assertEqual(reason_answer.accidental_input_count, 1)

    def test_back_chord_is_audited_and_replaces_later_answers(self) -> None:
        questionnaire = QuestionnaireFSM(timeout_s=30.0)
        questionnaire.start(_clock(0))
        questionnaire.handle_edges(("right.a",), _clock(1))
        questionnaire.handle_edges(("right.b",), _clock(2))  # q1 -> q2
        questionnaire.handle_edges(("left.x", "left.y"), _clock(3))  # q2 -> q1
        self.assertEqual(questionnaire.state, "q1")
        questionnaire.handle_edges(("left.y",), _clock(4))  # cursor: acceptable
        questionnaire.handle_edges(("right.a",), _clock(5))  # select
        questionnaire.handle_edges(("right.b",), _clock(6))  # confirm
        result = None
        for index in range(5):
            questionnaire.handle_edges(("right.a",), _clock(7 + index * 2))
            result = questionnaire.handle_edges(
                ("right.b",), _clock(8 + index * 2)
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.q1_response, "acceptable_as_is")
        self.assertEqual(result.rejection_reason_ids, ())
        self.assertEqual(result.back_correction_count, 1)
        self.assertEqual(len(result.answers), 6)
        answers = {answer.question_id: answer for answer in result.answers}
        corrected_q1 = answers["q1_modification_needed"]
        self.assertEqual(corrected_q1.prompt_shown_control_step, 3)
        self.assertEqual(corrected_q1.first_input_control_step, 4)
        self.assertEqual(corrected_q1.confirmed_control_step, 6)
        self.assertEqual(corrected_q1.response_latency_ms, 300.0)
        self.assertEqual(corrected_q1.back_correction_count, 1)
        self.assertEqual(answers["q2"].prompt_shown_control_step, 6)

    def test_main_questions_require_select_then_confirm(self) -> None:
        questionnaire = QuestionnaireFSM(timeout_s=30.0)
        questionnaire.start(_clock(0))
        self.assertIsNone(
            questionnaire.handle_edges(("right.b",), _clock(1))
        )
        self.assertEqual(questionnaire.state, "q1")
        self.assertIsNone(
            questionnaire.handle_edges(("right.a",), _clock(2))
        )
        self.assertEqual(questionnaire.state, "q1")
        questionnaire.handle_edges(("right.b",), _clock(3))
        self.assertEqual(questionnaire.state, "q2")

        q1 = questionnaire._answers["q1_modification_needed"]
        self.assertEqual(q1.prompt_shown_control_step, 0)
        self.assertEqual(q1.first_input_control_step, 1)
        self.assertEqual(q1.confirmed_control_step, 3)
        self.assertEqual(q1.response_latency_ms, 300.0)
        self.assertEqual(q1.input_device, "vr_controller")
        self.assertEqual(q1.back_correction_count, 0)
        self.assertEqual(q1.accidental_input_count, 1)

    def test_timeout_remains_missing_abstain(self) -> None:
        questionnaire = QuestionnaireFSM(timeout_s=1.0)
        questionnaire.start(_clock(0))
        result = questionnaire.handle_edges((), _clock(10))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.completion_status, "timed_out")
        self.assertEqual(result.response_disposition, "missing_abstain")
        self.assertIsNone(result.q1_response)
        self.assertIn("q1_modification_needed", result.missing_question_ids)
        self.assertEqual(len(result.answers), 1)
        missing = result.answers[0]
        self.assertEqual(missing.question_id, "q1_modification_needed")
        self.assertEqual(missing.answer_status, "unconfirmed")
        self.assertIsNone(missing.value)
        self.assertEqual(missing.prompt_shown_control_step, 0)
        self.assertEqual(missing.first_input_control_step, -1)
        self.assertEqual(missing.confirmed_control_step, -1)
        self.assertEqual(missing.response_latency_ms, -1.0)

    def test_timeout_preserves_current_prompt_and_partial_first_input(self) -> None:
        questionnaire = QuestionnaireFSM(timeout_s=1.0)
        questionnaire.start(_clock(0))
        questionnaire.handle_edges(("left.y",), _clock(1))
        result = questionnaire.handle_edges((), _clock(10))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.response_disposition, "missing_abstain")
        self.assertEqual(len(result.answers), 1)
        partial = result.answers[0]
        self.assertEqual(partial.answer_status, "unconfirmed")
        self.assertEqual(partial.prompt_shown_control_step, 0)
        self.assertEqual(partial.first_input_control_step, 1)
        self.assertEqual(partial.confirmed_control_step, -1)
        self.assertIsNone(partial.value)

    def test_uncertain_completion_remains_an_explicit_abstain(self) -> None:
        questionnaire = QuestionnaireFSM(timeout_s=30.0)
        questionnaire.start(_clock(0))
        questionnaire.handle_edges(("left.y",), _clock(1))
        questionnaire.handle_edges(("left.y",), _clock(2))
        questionnaire.handle_edges(("right.a",), _clock(3))
        questionnaire.handle_edges(("right.b",), _clock(4))
        result = None
        for index in range(5):
            questionnaire.handle_edges(("right.a",), _clock(5 + index * 2))
            result = questionnaire.handle_edges(
                ("right.b",), _clock(6 + index * 2)
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.completion_status, "completed")
        self.assertEqual(result.q1_response, "uncertain")
        self.assertEqual(result.response_disposition, "uncertain_abstain")

    def test_per_question_audit_validator_rejects_clock_latency_drift(self) -> None:
        query = {
            "query_id": "query-id",
            "trial_id": "trial-id",
            "response_status": "timeout",
            "issued_simulation_time_s": 0.0,
            "issued_monotonic_ns": 1_000_000_000,
            "issued_unix_ns": 2_000_000_000,
            "issued_control_step": 0,
            "completed_simulation_time_s": 0.5,
            "completed_monotonic_ns": 1_500_000_000,
            "completed_unix_ns": 2_500_000_000,
            "completed_control_step": 5,
            "back_correction_count": 1,
            "accidental_input_count": 1,
        }
        answer = {
            "query_id": "query-id",
            "trial_id": "trial-id",
            "question_id": ONLINE_Q1_ID,
            "answer_status": "confirmed",
            "value_json": '"uncertain"',
            "prompt_shown_simulation_time_s": 0.1,
            "prompt_shown_monotonic_ns": 1_100_000_000,
            "prompt_shown_unix_ns": 2_100_000_000,
            "prompt_shown_control_step": 1,
            "first_input_simulation_time_s": 0.2,
            "first_input_monotonic_ns": 1_200_000_000,
            "first_input_unix_ns": 2_200_000_000,
            "first_input_control_step": 2,
            "confirmed_simulation_time_s": 0.4,
            "confirmed_monotonic_ns": 1_400_000_000,
            "confirmed_unix_ns": 2_400_000_000,
            "confirmed_control_step": 4,
            "response_latency_ms": 300.0,
            "input_device": "vr_controller",
            "back_correction_count": 1,
            "accidental_input_count": 1,
        }
        partial = {
            "query_id": "query-id",
            "trial_id": "trial-id",
            "question_id": "q2_perceived_danger",
            "answer_status": "unconfirmed",
            "value_json": "null",
            "prompt_shown_simulation_time_s": 0.4,
            "prompt_shown_monotonic_ns": 1_400_000_000,
            "prompt_shown_unix_ns": 2_400_000_000,
            "prompt_shown_control_step": 4,
            "first_input_simulation_time_s": -1.0,
            "first_input_monotonic_ns": 0,
            "first_input_unix_ns": 0,
            "first_input_control_step": -1,
            "confirmed_simulation_time_s": -1.0,
            "confirmed_monotonic_ns": 0,
            "confirmed_unix_ns": 0,
            "confirmed_control_step": -1,
            "response_latency_ms": -1.0,
            "input_device": "vr_controller",
            "back_correction_count": 0,
            "accidental_input_count": 0,
        }

        def validator_for_audit() -> _Validator:
            validator = _Validator(
                config_path=DEFAULT_CONFIG,
                collection_path=None,
                participant_id=None,
                session_id=None,
                seed=None,
                mode=None,
                allow_partial_path=False,
            )
            validator.file = {"trials": None}
            return validator

        valid = validator_for_audit()
        valid._validate_query_answer_audit([query], [answer, partial])
        self.assertEqual(valid.issues, [])

        invalid = validator_for_audit()
        invalid._validate_query_answer_audit(
            [query], [{**answer, "response_latency_ms": 301.0}, partial]
        )
        self.assertIn(
            "feedback.answer_latency", {issue.code for issue in invalid.issues}
        )

        missing_partial = validator_for_audit()
        missing_partial._validate_query_answer_audit([query], [answer])
        self.assertIn(
            "feedback.answer_cardinality",
            {issue.code for issue in missing_partial.issues},
        )

        fabricated_partial = validator_for_audit()
        fabricated_partial._validate_query_answer_audit(
            [query], [answer, {**partial, "value_json": '"provisional"'}]
        )
        self.assertIn(
            "feedback.answer_value",
            {issue.code for issue in fabricated_partial.issues},
        )

    def test_emergency_abort_requires_two_button_hold_and_resets_on_disconnect(self) -> None:
        pressed = ControllerButtonSnapshot(
            left_connected=False,
            right_connected=True,
            right_a=True,
            right_b=True,
        )
        disconnected = ControllerButtonSnapshot(
            left_connected=False,
            right_connected=False,
        )
        monitor = EmergencyAbortMonitor(
            buttons=("right.a", "right.b"), hold_s=2.0
        )
        self.assertFalse(monitor.update(pressed, monotonic_ns=0))
        self.assertFalse(monitor.update(pressed, monotonic_ns=1_999_999_999))
        self.assertFalse(monitor.update(disconnected, monotonic_ns=2_000_000_000))
        self.assertFalse(monitor.update(pressed, monotonic_ns=3_000_000_000))
        self.assertTrue(monitor.update(pressed, monotonic_ns=5_000_000_000))
        self.assertFalse(monitor.update(pressed, monotonic_ns=6_000_000_000))


if __name__ == "__main__":
    unittest.main()
