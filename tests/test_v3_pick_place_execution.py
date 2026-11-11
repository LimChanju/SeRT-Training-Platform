from pathlib import Path

import pytest

from v3_chan.pick_place_execution import (
    GraspReadinessGate,
    failed_attempt_reaches_limit,
)


def test_grasp_gate_holds_until_alignment_is_stable():
    gate = GraspReadinessGate(
        max_distance_m=0.05,
        required_consecutive_steps=3,
        max_wait_steps=10,
    )

    assert not gate.update(
        controller_event=1,
        grasp_reference_position=[0.0, 0.0, 0.0],
        cube_position=[0.0, 0.0, 0.0],
    ).active

    far = gate.update(
        controller_event=2,
        grasp_reference_position=[0.2, 0.0, 0.0],
        cube_position=[0.0, 0.0, 0.0],
    )
    assert far.active
    assert not far.release
    assert not far.timed_out
    assert far.consecutive_ready_steps == 0

    for expected_steps in (1, 2):
        decision = gate.update(
            controller_event=2,
            grasp_reference_position=[0.02, 0.0, 0.0],
            cube_position=[0.0, 0.0, 0.0],
        )
        assert not decision.release
        assert decision.consecutive_ready_steps == expected_steps

    released = gate.update(
        controller_event=2,
        grasp_reference_position=[0.02, 0.0, 0.0],
        cube_position=[0.0, 0.0, 0.0],
    )
    assert released.release
    assert released.distance_m == pytest.approx(0.02)


def test_grasp_gate_releases_for_observed_controller_reference_pose():
    """Keep the normal Panda controller frame from regressing into a timeout."""

    gate = GraspReadinessGate(
        max_distance_m=0.070,
        required_consecutive_steps=3,
        max_wait_steps=180,
    )
    cube_position = [0.401, 0.201, 0.976]
    # panda_rightfinger pose observed at the normal event-2 settle pose.
    controller_reference_position = [0.401, 0.245, 1.022]

    for _ in range(2):
        decision = gate.update(
            controller_event=2,
            grasp_reference_position=controller_reference_position,
            cube_position=cube_position,
        )
        assert not decision.release

    released = gate.update(
        controller_event=2,
        grasp_reference_position=controller_reference_position,
        cube_position=cube_position,
    )
    assert released.release
    assert released.distance_m == pytest.approx(0.0636553219)


def test_grasp_gate_tolerates_observed_boundary_jitter():
    """A sub-millimeter pose deviation must not block an otherwise valid grasp."""

    gate = GraspReadinessGate(
        max_distance_m=0.075,
        required_consecutive_steps=3,
        max_wait_steps=180,
    )
    for _ in range(3):
        decision = gate.update(
            controller_event=2,
            grasp_reference_position=[0.0, 0.07018, 0.0],
            cube_position=[0.0, 0.0, 0.0],
        )
    assert decision.release
    assert decision.distance_m == pytest.approx(0.07018)


def test_grasp_gate_times_out_when_arm_never_reaches_cube():
    gate = GraspReadinessGate(
        max_distance_m=0.05,
        required_consecutive_steps=1,
        max_wait_steps=3,
    )

    for _ in range(2):
        decision = gate.update(
            controller_event=2,
            grasp_reference_position=[0.2, 0.0, 0.0],
            cube_position=[0.0, 0.0, 0.0],
        )
        assert not decision.timed_out

    decision = gate.update(
        controller_event=2,
        grasp_reference_position=[0.2, 0.0, 0.0],
        cube_position=[0.0, 0.0, 0.0],
    )
    assert decision.timed_out
    assert not decision.release
    assert decision.waited_steps == 3


def test_grasp_gate_resets_after_controller_leaves_settle_phase():
    gate = GraspReadinessGate(max_wait_steps=3)
    gate.update(
        controller_event=2,
        grasp_reference_position=[0.2, 0.0, 0.0],
        cube_position=[0.0, 0.0, 0.0],
    )
    reset = gate.update(
        controller_event=3,
        grasp_reference_position=[0.2, 0.0, 0.0],
        cube_position=[0.0, 0.0, 0.0],
    )
    assert not reset.active
    assert reset.waited_steps == 0


def test_failed_attempt_limit_is_inclusive():
    assert not failed_attempt_reaches_limit(
        current_cube_attempt=2, max_attempts_per_cube=3
    )
    assert failed_attempt_reaches_limit(
        current_cube_attempt=3, max_attempts_per_cube=3
    )


def test_failed_attempt_limit_rejects_invalid_values():
    with pytest.raises(ValueError):
        failed_attempt_reaches_limit(current_cube_attempt=0, max_attempts_per_cube=3)
    with pytest.raises(ValueError):
        failed_attempt_reaches_limit(current_cube_attempt=1, max_attempts_per_cube=0)


def test_hri_recorder_reuses_resolved_gripper_center():
    main_source = (
        Path(__file__).resolve().parents[1] / "v3_chan" / "main.py"
    ).read_text(encoding="utf-8")

    assert "gripper_center=gripper_center_for_distance" in main_source
    assert "gripper_center=_gripper_center_from_fingers(panda)" not in main_source
