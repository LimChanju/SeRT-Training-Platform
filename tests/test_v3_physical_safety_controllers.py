from types import SimpleNamespace

import numpy as np

from v3_chan.end_effector_safety_geometry import (
    EndEffectorSafetyResult,
    HandSafetyResult,
)
from v3_chan.physical_safety_controllers import (
    CBFConfig,
    DistalLinkVelocityCBF,
    project_velocity_qp,
)


def test_velocity_projection_satisfies_halfspace_and_joint_limits():
    projected, slack, feasible, before, after = project_velocity_qp(
        nominal_velocity=np.array([-1.0, 0.5]),
        constraint_matrix=np.array([[1.0, 0.0]]),
        lower_bounds=np.array([0.25]),
        velocity_lower=np.array([-2.0, -0.2]),
        velocity_upper=np.array([2.0, 0.2]),
    )

    np.testing.assert_allclose(projected, [0.25, 0.2], atol=1e-5)
    assert feasible
    assert np.isclose(slack, 0.0)
    assert before > 1.0
    assert after <= 1e-5


def test_velocity_projection_exposes_infeasible_constraints_with_slack():
    projected, slack, feasible, before, after = project_velocity_qp(
        nominal_velocity=np.array([0.0]),
        constraint_matrix=np.array([[1.0], [-1.0]]),
        lower_bounds=np.array([2.0, 2.0]),
        velocity_lower=np.array([-1.0]),
        velocity_upper=np.array([1.0]),
    )

    assert not feasible
    assert slack >= 1.99
    assert before >= 1.99
    assert after >= 1.99
    assert -1.0 <= projected[0] <= 1.0


class _FakeArticulationView:
    body_names = ("panda_link0", "panda_hand")

    def get_jacobians(self):
        jacobian = np.zeros((1, 1, 6, 1), dtype=float)
        jacobian[0, 0, 0, 0] = 1.0
        return jacobian


class _FakeRobot:
    def __init__(self):
        self._articulation_view = _FakeArticulationView()
        self.dof_properties = {
            "maxVelocity": np.array([2.0]),
            "lower": np.array([-3.0]),
            "upper": np.array([3.0]),
        }

    def get_joint_positions(self):
        return np.array([0.0])


class _FakeSafetyGeometry:
    def closest_link_world_pose(self, hand_result):
        assert hand_result.closest_link == "panda_hand"
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), True


def _dynamic_hand(*, velocity=None, closing_speed=0.0):
    return SimpleNamespace(
        hand_velocity_filtered_mps=np.zeros(3) if velocity is None else velocity,
        hand_velocity_valid=velocity is not None,
        closing_speed_mps=closing_speed,
        closing_speed_valid=closing_speed > 0.0,
    )


def test_cbf_changes_nominal_action_away_from_close_hand():
    left = HandSafetyResult(
        hand="left",
        geometry_valid=True,
        surface_gap_m=0.02,
        closest_link="panda_hand",
        closest_surface_point_world_pos=(0.0, 0.0, 0.0),
        closest_surface_point_valid=True,
        near=True,
        distance_gate=1.0,
    )
    safety_result = EndEffectorSafetyResult(
        left=left,
        right=HandSafetyResult(hand="right", geometry_valid=False),
    )
    dynamic_sample = SimpleNamespace(
        left=_dynamic_hand(),
        right=_dynamic_hand(),
    )
    action = SimpleNamespace(
        joint_indices=np.array([0]),
        joint_positions=np.array([-0.01]),
        joint_velocities=None,
    )
    cbf = DistalLinkVelocityCBF(
        CBFConfig(
            safe_gap_m=0.05,
            activation_gap_m=0.13,
            gamma_per_s=8.0,
            prediction_horizon_s=0.0,
        )
    )

    filtered, diagnostics = cbf.filter_action(
        robot=_FakeRobot(),
        arm_action=action,
        safety_result=safety_result,
        dynamic_sample=dynamic_sample,
        safety_geometry=_FakeSafetyGeometry(),
        observation={
            "human_left_hand_pos": np.array([-0.10, 0.0, 0.0]),
            "human_right_hand_pos": np.zeros(3),
        },
        physics_dt_s=0.1,
    )

    # h = 0.02 - 0.05, so qdot >= -gamma*h = 0.24 rad/s.
    assert filtered.joint_velocities[0] >= 0.24 - 1e-5
    assert filtered.joint_positions[0] > 0.0
    assert diagnostics.active
    assert diagnostics.feasible
    assert diagnostics.constraint_count == 1
    assert diagnostics.intervention_norm_radps > 0.3
    assert diagnostics.max_constraint_violation_after <= 1e-5


def test_cbf_stays_inactive_when_hand_is_outside_activation_gap():
    safety_result = EndEffectorSafetyResult(
        left=HandSafetyResult(
            hand="left",
            geometry_valid=True,
            surface_gap_m=0.20,
            closest_link="panda_hand",
            closest_surface_point_world_pos=(0.0, 0.0, 0.0),
            closest_surface_point_valid=True,
        ),
        right=HandSafetyResult(hand="right", geometry_valid=False),
    )
    action = SimpleNamespace(
        joint_indices=np.array([0]),
        joint_positions=np.array([-0.01]),
        joint_velocities=None,
    )
    cbf = DistalLinkVelocityCBF()

    filtered, diagnostics = cbf.filter_action(
        robot=_FakeRobot(),
        arm_action=action,
        safety_result=safety_result,
        dynamic_sample=SimpleNamespace(
            left=_dynamic_hand(),
            right=_dynamic_hand(),
        ),
        safety_geometry=_FakeSafetyGeometry(),
        observation={
            "human_left_hand_pos": np.array([-0.25, 0.0, 0.0]),
            "human_right_hand_pos": np.zeros(3),
        },
        physics_dt_s=0.1,
    )

    assert not diagnostics.active
    assert diagnostics.constraint_count == 0
    assert np.isclose(filtered.joint_velocities[0], -0.1)
    assert np.isclose(filtered.joint_positions[0], -0.01)
