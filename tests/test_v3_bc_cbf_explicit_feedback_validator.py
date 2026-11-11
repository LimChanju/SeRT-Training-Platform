"""Fail-closed validation tests for the BC+CBF explicit-feedback artifact.

The fixture is deliberately written through ``ExplicitFeedbackRecorder``.  This
keeps the test corpus coupled to the collection writer's real row schema instead
of maintaining a second hand-written HDF5 schema in the tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from v3_chan.explicit_feedback.recorder import (
    ExplicitFeedbackRecorder,
    _row_specs,
)
from v3_chan.explicit_feedback.policy import mask_human_observations
from v3_chan.explicit_feedback.schema import (
    COLLECTION_SCHEMA_VERSION,
    POLICY_SHA256,
    QUESTION_ID,
    RUNTIME_CONTRACT_SHA256,
    EncounterRecord,
    PromptRecord,
    ResponseRecord,
)
from v3_chan.explicit_feedback.validator import (
    CollectionValidationError,
    validate_collection,
)
from v3_chan.rl.observations import observation_slices
from v3_chan.validate_bc_cbf_explicit_feedback import main as validator_main


JOINT_NAMES = tuple(
    [f"panda_joint{index}" for index in range(1, 8)]
    + ["panda_finger_joint1", "panda_finger_joint2"]
)
ARM_JOINT_INDICES = tuple(range(7))
EPISODE = "episodes/episode_000000"


def _blank_row() -> dict[str, object]:
    row: dict[str, object] = {}
    for path, spec in _row_specs(len(JOINT_NAMES), len(ARM_JOINT_INDICES)).items():
        if spec.dtype == "str":
            row[path] = ""
            continue
        value = np.zeros(spec.tail_shape, dtype=spec.dtype)
        row[path] = value.item() if spec.tail_shape == () else value
    return row


def _set_action_channel(
    row: dict[str, object],
    *,
    stage: str,
    channel: str,
    unit: str,
    values: np.ndarray,
    mask: np.ndarray,
) -> None:
    base = f"actions/{stage}_{channel}_{unit}"
    row[base] = np.asarray(values, dtype=np.float64)
    row[base + "_mask"] = np.asarray(mask, dtype=np.int8)


def _make_row(step: int) -> dict[str, object]:
    """Return one internally consistent transition.

    Row zero submits the complete filtered arm action.  Row one is the frozen
    gripper arbitration behavior: only finger targets are submitted and the arm
    drive targets are carried from pre-applied to post-applied unchanged.
    """

    row = _blank_row()
    is_gripper_row = step == 1
    observation_ns = 100 + 100 * step
    q_before = np.zeros(9, dtype=np.float64)
    if is_gripper_row:
        q_before[0] = 0.001
    qdot_nominal = np.zeros(9, dtype=np.float64)
    qdot_nominal[0] = 0.2 if step == 0 else 0.1
    qdot_filtered = qdot_nominal.copy()
    if step == 0:
        qdot_filtered[0] = 0.1

    positions = {
        "head": np.asarray([0.50, 0.0, 1.5]),
        "left": np.asarray([0.10 if step == 0 else 0.20, 0.0, 0.8]),
        "right": np.asarray([0.20, 0.0, 0.8]),
    }
    slices = observation_slices()
    raw_observation = np.zeros(84, dtype=np.float32)
    next_observation = np.zeros(84, dtype=np.float32)
    for entity, field in (
        ("head", "human_head_pos"),
        ("left", "human_left_hand_pos"),
        ("right", "human_right_hand_pos"),
    ):
        raw_observation[slices[field]] = positions[entity]
        next_observation[slices[field]] = positions[entity]
    minimum_gap = 0.10 if step == 0 else 0.20
    raw_observation[slices["min_hand_gripper_dist"]] = minimum_gap
    raw_observation[slices["human_robot_collision"]] = 0.0
    raw_observation[slices["near_human"]] = 1.0 if step == 0 else 0.0

    row.update(
        {
            "control/source_step": step,
            "control/result_step": step + 1,
            "control/sim_time": 0.01 * step,
            "control/observation_monotonic_ns": observation_ns,
            "control/preflight_monotonic_ns": observation_ns + 5,
            "control/wall_time_unix_ns": 1_100 + 100 * step,
            "control/policy_inference_started_monotonic_ns": observation_ns + 8,
            "control/policy_output_monotonic_ns": observation_ns + 10,
            "control/rmpflow_output_monotonic_ns": observation_ns + 20,
            "control/cbf_output_monotonic_ns": observation_ns + 30,
            "control/action_command_monotonic_ns": observation_ns + 40,
            "control/next_observation_monotonic_ns": observation_ns + 45,
            "control/post_step_monotonic_ns": observation_ns + 50,
            "observations/obs_raw": raw_observation,
            "observations/obs_policy": mask_human_observations(
                raw_observation
            ),
            "observations/obs_next": next_observation,
            "human/valid_mask": np.ones(3, dtype=np.int8),
            "human_next/valid_mask": np.ones(3, dtype=np.int8),
            "safety/left_surface_gap_m": 0.10 if step == 0 else 0.20,
            "safety/right_surface_gap_m": 0.20,
            "safety/min_surface_gap_m": 0.10 if step == 0 else 0.20,
            "safety/geometry_valid": 1,
            "safety/collision": 0,
            "safety/near": 1 if step == 0 else 0,
            "safety/near_miss": 0,
            "actions/bc_task_action": np.zeros(5, dtype=np.float32),
            "actions/arm_action_submitted": 0 if is_gripper_row else 1,
            "actions/action_pipeline_complete": 1,
            "actions/gripper_command": "open" if is_gripper_row else "none",
            "robot/measured_joint_positions_before_rad": q_before,
            "robot/measured_joint_velocities_before_rad_s": (
                np.asarray([0.1] + [0.0] * 8) if is_gripper_row else np.zeros(9)
            ),
            "task/controller_event_before": 6 if is_gripper_row else 5,
            "task/controller_event_after": 7 if is_gripper_row else 5,
            "task/controller_t_before": 1.0 if is_gripper_row else 0.2,
            "task/controller_t_after": 0.0 if is_gripper_row else 0.21,
            "task/success": 1 if is_gripper_row else 0,
            "task/terminated": 1 if is_gripper_row else 0,
            "task/truncated": 0,
            "task/terminal_reason": "success" if is_gripper_row else "",
            "encounter/row_encounter_id": "encounter-1",
            "encounter/state": "robot_hold" if is_gripper_row else "encounter_active",
        }
    )

    for namespace in ("human", "human_next"):
        acquisition_ns = (
            observation_ns - 10
            if namespace == "human"
            else observation_ns + 35
        )
        for entity, position in positions.items():
            row[f"{namespace}/{entity}_position_world_m"] = position
            row[f"{namespace}/{entity}_pose_valid"] = 1
            row[f"{namespace}/{entity}_position_tracked"] = 1
            row[f"{namespace}/{entity}_tracking_status_known"] = 1
            row[f"{namespace}/{entity}_source_name"] = (
                "hmd_xr_physical" if entity == "head" else "xr_physical"
            )
            row[f"{namespace}/{entity}_source_path"] = f"/World/XR/{entity}"
            row[f"{namespace}/{entity}_acquisition_monotonic_ns"] = (
                acquisition_ns
            )
            row[f"{namespace}/{entity}_pose_age_ms"] = 10.0 / 1e6
            row[f"{namespace}/{entity}_source_switched"] = 0

    arm_mask = np.asarray([1] * 7 + [0, 0], dtype=np.int8)
    none_mask = np.zeros(9, dtype=np.int8)
    full_mask = np.ones(9, dtype=np.int8)
    nominal_positions = q_before + 0.01 * qdot_nominal
    filtered_positions = q_before + 0.01 * qdot_filtered
    for stage, target_positions, target_velocities in (
        ("nominal_rmpflow", nominal_positions, qdot_nominal),
        ("cbf_filtered", filtered_positions, qdot_filtered),
    ):
        _set_action_channel(
            row,
            stage=stage,
            channel="joint_positions",
            unit="rad",
            values=target_positions,
            mask=arm_mask,
        )
        _set_action_channel(
            row,
            stage=stage,
            channel="joint_velocities",
            unit="rad_s",
            values=target_velocities,
            mask=arm_mask,
        )
        _set_action_channel(
            row,
            stage=stage,
            channel="joint_efforts",
            unit="nm",
            values=np.zeros(9),
            mask=none_mask,
        )

    if is_gripper_row:
        submitted_positions = np.zeros(9)
        submitted_positions[-2:] = 0.04
        submitted_position_mask = np.asarray([0] * 7 + [1, 1], dtype=np.int8)
        submitted_velocities = np.zeros(9)
        submitted_velocity_mask = none_mask
    else:
        submitted_positions = filtered_positions.copy()
        submitted_position_mask = arm_mask
        submitted_velocities = qdot_filtered.copy()
        submitted_velocity_mask = arm_mask
    _set_action_channel(
        row,
        stage="submitted",
        channel="joint_positions",
        unit="rad",
        values=submitted_positions,
        mask=submitted_position_mask,
    )
    _set_action_channel(
        row,
        stage="submitted",
        channel="joint_velocities",
        unit="rad_s",
        values=submitted_velocities,
        mask=submitted_velocity_mask,
    )
    _set_action_channel(
        row,
        stage="submitted",
        channel="joint_efforts",
        unit="nm",
        values=np.zeros(9),
        mask=none_mask,
    )

    if is_gripper_row:
        pre_positions = np.zeros(9)
        pre_positions[0] = 0.001
        pre_velocities = np.zeros(9)
        pre_velocities[0] = 0.1
    else:
        pre_positions = np.zeros(9)
        pre_velocities = np.zeros(9)
    applied_positions = pre_positions.copy()
    applied_velocities = pre_velocities.copy()
    if is_gripper_row:
        applied_positions[-2:] = 0.04
    else:
        applied_positions[:7] = filtered_positions[:7]
        applied_velocities[:7] = qdot_filtered[:7]

    for stage, target_positions, target_velocities in (
        ("pre_applied", pre_positions, pre_velocities),
        ("applied", applied_positions, applied_velocities),
    ):
        _set_action_channel(
            row,
            stage=stage,
            channel="joint_positions",
            unit="rad",
            values=target_positions,
            mask=full_mask,
        )
        _set_action_channel(
            row,
            stage=stage,
            channel="joint_velocities",
            unit="rad_s",
            values=target_velocities,
            mask=full_mask,
        )
        _set_action_channel(
            row,
            stage=stage,
            channel="joint_efforts",
            unit="nm",
            values=np.zeros(9),
            mask=none_mask,
        )

    row["actions/applied_arm_joint_velocities_rad_s"] = applied_velocities[:7]
    row["actions/applied_arm_joint_velocities_mask"] = np.ones(7, dtype=np.int8)
    measured_after = q_before.copy()
    if is_gripper_row:
        measured_after[-2:] = 0.04
    else:
        measured_after[0] = 0.001
    row["robot/measured_joint_positions_after_rad"] = measured_after
    row["robot/measured_joint_velocities_after_rad_s"] = applied_velocities

    row.update(
        {
            "cbf/intervention_norm_radps": float(
                np.linalg.norm(qdot_filtered[:7] - qdot_nominal[:7])
            ),
            "cbf/nominal_velocity_norm_radps": float(
                np.linalg.norm(qdot_nominal[:7])
            ),
            "cbf/filtered_velocity_norm_radps": float(
                np.linalg.norm(qdot_filtered[:7])
            ),
            "cbf/max_constraint_violation_before_mps": 0.01 if step == 0 else 0.0,
            "cbf/max_constraint_violation_after_mps": 0.0,
            "cbf/slack_mps": 0.0,
            "cbf/buffered_current_gap_m": 0.10 if step == 0 else 10.0,
            "cbf/solve_time_ms": 0.2 if step == 0 else 0.0,
            "cbf/active": 1 if step == 0 else 0,
            "cbf/intervention_available": 1,
            "cbf/constraint_count": 1 if step == 0 else 0,
            "cbf/tracked_hand_count": 2,
            "cbf/valid_hand_count": 2,
            "cbf/feasible": 1,
            "cbf/fallback_applied": 0,
            "cbf/solver_converged": 1,
            "cbf/infeasibility_proven": 0,
            "cbf/relaxed_solution_available": 0,
            "cbf/relaxed_solution_applied": 0,
            "cbf/intentional_human_absence": 0,
            "cbf/failure_reasons_json": "[]",
            "cbf/status": "solved" if step == 0 else "inactive",
            "cbf/solver_backend": "test_qp" if step == 0 else "none",
            "cbf/projection_status": "optimal" if step == 0 else "inactive",
        }
    )
    return row


def _create_valid_collection(
    directory: Path,
    *,
    name: str = "collection.h5",
    finalize: bool = True,
) -> Path:
    output = directory / name
    recorder = ExplicitFeedbackRecorder(
        output,
        metadata={
            "participant_id": "P001",
            "session_id": "session-001",
            "code_version": "3682056fd8888d2881290f1e1f65f3cce10270d9",
            "source_tree_sha256": "a" * 64,
            "physics_dt_s": 0.01,
            "max_pose_age_ms": 100.0,
            "prompt_timeout_s": 1.0,
            "session_start_monotonic_ns": 50,
            "session_start_unix_ns": 1_000,
        },
        joint_names=JOINT_NAMES,
        arm_joint_indices=ARM_JOINT_INDICES,
    )
    recorder.start_episode(
        0,
        seed=7,
        start_monotonic_ns=90,
        start_unix_ns=1_090,
        initial_scene={"fixture_id": "validator-test-v1"},
    )
    recorder.append_transition(_make_row(0))
    recorder.append_transition(_make_row(1))
    recorder.end_episode(
        success=True,
        terminated=True,
        truncated=False,
        terminal_reason="success",
        end_monotonic_ns=300,
        end_unix_ns=1_300,
    )
    recorder.append_encounter(
        EncounterRecord(
            encounter_id="encounter-1",
            episode_index=0,
            start_step=0,
            end_step_exclusive=2,
            # Encounter classification is available at each row's CBF output.
            start_monotonic_ns=130,
            end_monotonic_ns=230,
            minimum_surface_gap_m=0.10,
            involved_hand_mask=(1, 0),
            eligible=True,
            eligibility_reason="cbf_activation_band_entry",
            terminal_status="completed",
            prompt_id="prompt-1",
        )
    )
    recorder.append_prompt(
        PromptRecord(
            prompt_id="prompt-1",
            encounter_id="encounter-1",
            question_id=QUESTION_ID,
            issued_monotonic_ns=260,
            issued_unix_ns=1_260,
            presentation_deadline_monotonic_ns=1_000_000_260,
            presented_monotonic_ns=270,
            deadline_monotonic_ns=1_000_000_270,
            status="accepted",
        )
    )
    recorder.append_response(
        ResponseRecord(
            response_id="response-1",
            client_response_id="client-response-1",
            prompt_id="prompt-1",
            response_code="acceptable",
            source="keyboard",
            client_unix_ns=1_280,
            received_monotonic_ns=280,
            received_unix_ns=1_280,
            accepted=True,
            rejection_reason="",
        )
    )
    partial = recorder.seal()
    return recorder.commit_validated() if finalize else partial


def _issue_codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def _assert_invalid(path: Path, code: str, *, allow_partial_path: bool = False):
    report = validate_collection(
        path,
        allow_partial_path=allow_partial_path,
        raise_on_error=False,
    )
    assert report.valid is False
    assert report.study_eligible is False
    assert code in _issue_codes(report), [
        (issue.code, issue.message) for issue in report.issues
    ]
    return report


def _append_duplicate_accepted_response(data: h5py.File) -> None:
    responses = data["explicit_feedback/responses"]
    for dataset in responses.values():
        dataset.resize((2, *dataset.shape[1:]))
        dataset[1] = dataset[0]
    responses["response_id"][1] = "response-2"
    data.attrs["response_count"] = 2
    data[f"{EPISODE}"].attrs["accepted_response_count"] = 2


def _append_rejected_duplicate_client_response(data: h5py.File) -> None:
    responses = data["explicit_feedback/responses"]
    for dataset in responses.values():
        dataset.resize((2, *dataset.shape[1:]))
        dataset[1] = dataset[0]
    responses["response_id"][1] = "response-2"
    responses["accepted"][1] = 0
    responses["rejection_reason"][1] = "duplicate_response_id"
    data.attrs["response_count"] = 2


def _append_rejected_response(
    data: h5py.File,
    *,
    prompt_id: str,
    rejection_reason: str,
    client_response_id: str = "rejected-client-response",
    response_code: str = "acceptable",
    received_monotonic_ns: int | None = None,
) -> None:
    responses = data["explicit_feedback/responses"]
    index = int(responses["response_id"].shape[0])
    for dataset in responses.values():
        dataset.resize((index + 1, *dataset.shape[1:]))
        dataset[index] = dataset[0]
    responses["response_id"][index] = f"rejected-server-response-{index}"
    responses["client_response_id"][index] = client_response_id
    responses["prompt_id"][index] = prompt_id
    responses["response_code"][index] = response_code
    responses["received_monotonic_ns"][index] = (
        281 + index
        if received_monotonic_ns is None
        else received_monotonic_ns
    )
    responses["received_unix_ns"][index] = 1_281 + index
    responses["accepted"][index] = 0
    responses["rejection_reason"][index] = rejection_reason
    data.attrs["response_count"] = index + 1


def test_validator_accepts_complete_collection_and_sparse_gripper_carry_forward(
    tmp_path,
):
    path = _create_valid_collection(tmp_path)

    report = validate_collection(path)

    assert report.valid is True
    assert report.study_eligible is True
    assert tuple(report.issues) == ()
    with h5py.File(path, "r") as data:
        assert data.attrs["schema_version"] == COLLECTION_SCHEMA_VERSION
        assert data.attrs["runtime_contract_sha256"] == RUNTIME_CONTRACT_SHA256
        assert data.attrs["policy_sha256"] == POLICY_SHA256
        assert data[f"{EPISODE}/actions/arm_action_submitted"][1] == 0
        assert not np.any(
            data[f"{EPISODE}/actions/submitted_joint_positions_rad_mask"][1, :7]
        )


def test_validator_accepts_a_consistent_completed_failed_task(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        episode = data[EPISODE]
        episode.attrs["success"] = 0
        episode.attrs["terminated"] = 0
        episode.attrs["truncated"] = 1
        episode.attrs["terminal_reason"] = "time_limit"
        episode["task/success"][1] = 0
        episode["task/terminated"][1] = 0
        episode["task/truncated"][1] = 1
        episode["task/terminal_reason"][1] = "time_limit"

    report = validate_collection(path)

    assert report.valid is True
    assert report.study_eligible is True


def test_validator_accepts_physical_xr_pose_when_tracking_flag_is_unavailable(
    tmp_path,
):
    """SteamVR physical poses are live/valid but do not expose TRACKED_BIT."""

    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        for entity in ("head", "left", "right"):
            data[f"{EPISODE}/human/{entity}_tracking_status_known"][:] = 0
            data[f"{EPISODE}/human/{entity}_position_tracked"][:] = -1

    report = validate_collection(path)

    assert report.valid is True
    assert report.study_eligible is True


def test_partial_path_is_publicly_rejected_but_internal_precommit_validation_is_strict(
    tmp_path, capsys
):
    partial = _create_valid_collection(tmp_path, finalize=False)

    _assert_invalid(partial, "file.partial_path")
    internal = validate_collection(partial, allow_partial_path=True)
    assert internal.valid is True
    assert internal.study_eligible is True

    exit_code = validator_main([str(partial), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["valid"] is False
    assert payload["study_eligible"] is False
    assert "file.partial_path" in {issue["code"] for issue in payload["issues"]}


@pytest.mark.parametrize(
    ("attribute", "bad_value"),
    (
        ("schema_version", "legacy_collection_v0"),
        ("runtime_contract_sha256", "0" * 64),
        ("policy_sha256", "f" * 64),
    ),
)
def test_root_contract_mismatch_is_rejected(tmp_path, attribute, bad_value):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data.attrs[attribute] = bad_value

    _assert_invalid(path, "root.attribute_mismatch")


def test_missing_required_row_dataset_is_rejected(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        del data[f"{EPISODE}/observations/obs_next"]

    _assert_invalid(path, "episode.dataset_missing")


def test_wrong_required_row_dataset_shape_is_rejected(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        parent = data[f"{EPISODE}/observations"]
        del parent["obs_next"]
        parent.create_dataset("obs_next", data=np.zeros((2, 83), dtype=np.float32))

    _assert_invalid(path, "episode.dataset_shape")


def test_policy_input_must_equal_the_frozen_human_observation_mask(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data[f"{EPISODE}/observations/obs_policy"][0, 0] = 1.0

    _assert_invalid(path, "policy.human_mask_mismatch")


def test_action_pipeline_incomplete_is_rejected(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data[f"{EPISODE}/actions/action_pipeline_complete"][0] = 0

    _assert_invalid(path, "action.pipeline_incomplete")


def test_sparse_gripper_arm_target_carry_forward_mismatch_is_rejected(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data[f"{EPISODE}/actions/applied_joint_positions_rad"][1, 0] += 0.25

    _assert_invalid(path, "action.sparse_gripper_carry_forward")


def test_filtered_velocity_norm_diagnostic_mismatch_is_rejected(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data[f"{EPISODE}/cbf/filtered_velocity_norm_radps"][0] = 99.0

    _assert_invalid(path, "action.filtered_norm_mismatch")


def test_inactive_cbf_must_preserve_the_nominal_arm_command(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data[f"{EPISODE}/actions/cbf_filtered_joint_velocities_rad_s"][1, 0] = 0.5

    _assert_invalid(path, "action.inactive_inconsistent")


@pytest.mark.parametrize(
    ("dataset", "value"),
    (
        ("fallback_applied", 1),
        ("feasible", 0),
        ("solver_converged", 0),
        ("relaxed_solution_applied", 1),
        ("intentional_human_absence", 1),
    ),
)
def test_unsafe_cbf_diagnostic_is_fail_closed(tmp_path, dataset, value):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data[f"{EPISODE}/cbf/{dataset}"][0] = value

    _assert_invalid(path, "cbf.unsafe_diagnostics")


@pytest.mark.parametrize(
    "dataset",
    (
        "human/valid_mask",
        "human/left_pose_valid",
        "human/left_position_tracked",
        "human/left_source_switched",
    ),
)
def test_tracking_fault_during_eligible_encounter_is_rejected(tmp_path, dataset):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        target = data[f"{EPISODE}/{dataset}"]
        if dataset == "human/valid_mask":
            target[0, 1] = 0
        elif dataset.endswith("source_switched"):
            target[0] = 1
        else:
            target[0] = 0

    _assert_invalid(path, "tracking.invalid_during_eligible_encounter")


def test_invalid_terminal_next_observation_tracking_is_rejected(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data[f"{EPISODE}/human_next/right_pose_valid"][-1] = 0
        data[f"{EPISODE}/human_next/valid_mask"][-1, 2] = 0

    _assert_invalid(path, "tracking.invalid_row")


def test_invalid_human_robot_geometry_is_rejected(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data[f"{EPISODE}/safety/geometry_valid"][0] = 0

    _assert_invalid(path, "safety.geometry_invalid")


def test_stored_minimum_surface_gap_must_match_per_hand_gaps(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data[f"{EPISODE}/safety/min_surface_gap_m"][0] = 0.12

    _assert_invalid(path, "safety.minimum_gap_mismatch")


def test_human_next_provenance_uses_its_exact_observation_clock(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        next_sample = int(
            data[f"{EPISODE}/control/next_observation_monotonic_ns"][0]
        )
        # Still precedes post_step, but it is later than the exact snapshot
        # clock and therefore cannot be provenance for obs_next.
        data[
            f"{EPISODE}/human_next/right_acquisition_monotonic_ns"
        ][0] = next_sample + 1

    _assert_invalid(path, "tracking.invalid_row")


@pytest.mark.parametrize(
    ("observation_path", "namespace"),
    (
        ("observations/obs_raw", "human"),
        ("observations/obs_next", "human_next"),
    ),
)
def test_observation_human_pose_must_match_its_tracking_provenance(
    tmp_path,
    observation_path,
    namespace,
):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        left_slice = observation_slices()["human_left_hand_pos"]
        data[f"{EPISODE}/{observation_path}"][0, left_slice.start] += 0.01
        # Prove the failure is the provenance relation, not the frozen policy
        # mask (human coordinates are masked from the BC input).
        if namespace == "human":
            raw = data[f"{EPISODE}/observations/obs_raw"][0]
            data[f"{EPISODE}/observations/obs_policy"][0] = (
                mask_human_observations(raw)
            )

    _assert_invalid(path, "tracking.observation_mismatch")


def test_raw_observation_safety_fields_must_match_serialized_diagnostics(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        near_slice = observation_slices()["near_human"]
        data[f"{EPISODE}/observations/obs_raw"][0, near_slice.start] = 0.0
        raw = data[f"{EPISODE}/observations/obs_raw"][0]
        data[f"{EPISODE}/observations/obs_policy"][0] = (
            mask_human_observations(raw)
        )

    _assert_invalid(path, "safety.observation_mismatch")


def test_prompt_must_be_issued_after_the_encounter_closes(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data["explicit_feedback/prompts/issued_monotonic_ns"][0] = 229

    _assert_invalid(path, "prompt.temporal_order")


def test_accepted_response_must_arrive_before_deadline(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        deadline = data["explicit_feedback/prompts/deadline_monotonic_ns"][0]
        data["explicit_feedback/responses/received_monotonic_ns"][0] = deadline

    _assert_invalid(path, "response.temporal_order")


@pytest.mark.parametrize("timeout_s", [np.nan, 0.09, 3600.1])
def test_prompt_timeout_root_attribute_has_closed_finite_bounds(
    tmp_path, timeout_s
):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data.attrs["prompt_timeout_s"] = timeout_s

    _assert_invalid(path, "root.attribute_mismatch")


def test_prompt_presentation_window_must_exactly_match_timeout(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data["explicit_feedback/prompts/presentation_deadline_monotonic_ns"][0] += 1

    _assert_invalid(path, "prompt.timeout_window")


def test_prompt_response_window_must_exactly_match_timeout(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data["explicit_feedback/prompts/deadline_monotonic_ns"][0] += 1

    _assert_invalid(path, "prompt.timeout_window")


def test_prompt_must_be_presented_before_presentation_deadline(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        deadline = data[
            "explicit_feedback/prompts/presentation_deadline_monotonic_ns"
        ][0]
        data["explicit_feedback/prompts/presented_monotonic_ns"][0] = deadline
        data["explicit_feedback/prompts/deadline_monotonic_ns"][0] = (
            deadline + 1_000_000_000
        )

    _assert_invalid(path, "prompt.temporal_order")


def test_response_cannot_reference_an_unknown_prompt(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data["explicit_feedback/responses/prompt_id"][0] = "missing-prompt"

    _assert_invalid(path, "response.orphan_prompt")


def test_each_prompt_has_exactly_one_accepted_response(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        _append_duplicate_accepted_response(data)

    _assert_invalid(path, "response.accepted_count")


def test_rejected_client_retry_is_retained_without_duplicate_server_row_id(
    tmp_path,
):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        _append_rejected_duplicate_client_response(data)

    report = validate_collection(path)

    assert report.valid is True
    assert report.response_count == 2


@pytest.mark.parametrize("rejection_reason", ("unknown_prompt_id", "broker_closed"))
def test_rejected_unknown_prompt_attempt_is_a_valid_audit_row(
    tmp_path,
    rejection_reason,
):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        _append_rejected_response(
            data,
            prompt_id="not-a-published-prompt",
            rejection_reason=rejection_reason,
        )

    report = validate_collection(path)

    assert report.valid is True
    assert report.response_count == 2


def test_broker_closed_has_priority_over_duplicate_and_unknown_checks(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        _append_rejected_response(
            data,
            prompt_id="not-a-published-prompt",
            rejection_reason="broker_closed",
            client_response_id="client-response-1",
        )

    report = validate_collection(path)

    assert report.valid is True
    assert report.response_count == 2


def test_late_reanswer_to_accepted_prompt_is_retained_as_rejected_audit(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        deadline = int(
            data["explicit_feedback/prompts/deadline_monotonic_ns"][0]
        )
        _append_rejected_response(
            data,
            prompt_id="prompt-1",
            rejection_reason="prompt_already_answered",
            received_monotonic_ns=deadline + 1,
        )

    report = validate_collection(path)

    assert report.valid is True
    assert report.response_count == 2


@pytest.mark.parametrize(
    ("dataset", "bad_value"),
    (("source", ""), ("received_monotonic_ns", 0), ("received_unix_ns", 0)),
)
def test_rejected_response_still_requires_server_audit_fields(
    tmp_path,
    dataset,
    bad_value,
):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        _append_rejected_response(
            data,
            prompt_id="not-a-published-prompt",
            rejection_reason="unknown_prompt_id",
        )
        data[f"explicit_feedback/responses/{dataset}"][1] = bad_value

    _assert_invalid(path, "response.audit_fields")


def test_unknown_prompt_with_inconsistent_rejection_reason_is_rejected(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        _append_rejected_response(
            data,
            prompt_id="not-a-published-prompt",
            rejection_reason="invalid_response_code",
            response_code="not-a-protocol-code",
        )

    _assert_invalid(path, "response.orphan_prompt")


def test_legacy_errp_group_is_never_a_feedback_channel(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data.create_group("errp")

    _assert_invalid(path, "root.forbidden_errp")


def test_disguised_case_variant_errp_group_is_also_forbidden(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data.create_group("pseudo_ErrP_predictions")

    _assert_invalid(path, "root.forbidden_errp")


def test_raise_on_error_exposes_a_dedicated_validation_exception(tmp_path):
    path = _create_valid_collection(tmp_path)
    with h5py.File(path, "r+") as data:
        data[f"{EPISODE}/actions/action_pipeline_complete"][0] = 0

    with pytest.raises(CollectionValidationError) as raised:
        validate_collection(path)
    assert "action.pipeline_incomplete" in _issue_codes(raised.value.report)
