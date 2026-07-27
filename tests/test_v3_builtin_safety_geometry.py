import math
import sys
import types

import numpy as np

from v3_chan.end_effector_safety_geometry import (
    DISTAL_LINK_NAMES,
    LINK_ID_BY_NAME,
    EndEffectorSafetyResult,
    HandSafetyResult,
    SafetyThresholds,
    classify_surface_gap,
    distance_gate,
)
from v3_chan.end_effector_safety_runtime import PandaEndEffectorSafetyRuntime


def test_distal_link_schema_includes_requested_franka_links():
    assert DISTAL_LINK_NAMES == (
        "panda_link6",
        "panda_link7",
        "panda_link8",
        "panda_hand",
        "panda_leftfinger",
        "panda_rightfinger",
    )
    assert len(set(LINK_ID_BY_NAME.values())) == len(DISTAL_LINK_NAMES)


def test_surface_gap_classification_boundaries():
    thresholds = SafetyThresholds().validated()

    collision = classify_surface_gap(0.0, thresholds)
    assert collision.collision
    assert collision.near
    assert not collision.near_miss

    near_miss = classify_surface_gap(0.02, thresholds)
    assert not near_miss.collision
    assert near_miss.near_miss
    assert near_miss.near

    near = classify_surface_gap(0.05, thresholds)
    assert not near.collision
    assert not near.near_miss
    assert near.near

    far = classify_surface_gap(0.050001, thresholds)
    assert not far.collision
    assert not far.near_miss
    assert not far.near

    invalid = classify_surface_gap(-1.0, thresholds, geometry_valid=False)
    assert not invalid.collision
    assert not invalid.near_miss
    assert not invalid.near
    assert invalid.distance_gate == 0.0


def test_distance_gate_is_clipped_linear_surface_gap_gate():
    thresholds = SafetyThresholds().validated()
    assert distance_gate(0.14, thresholds) == 0.0
    assert distance_gate(0.13, thresholds) == 0.0
    assert np.isclose(distance_gate(0.09, thresholds), 0.5)
    assert distance_gate(0.05, thresholds) == 1.0
    assert distance_gate(-0.01, thresholds) == 1.0
    assert distance_gate(math.inf, thresholds) == 0.0


def test_two_hand_aggregation_uses_minimum_gap_and_maximum_gate():
    left = HandSafetyResult(
        hand="left",
        geometry_valid=True,
        surface_gap_m=0.08,
        near=False,
        distance_gate=0.625,
    )
    right = HandSafetyResult(
        hand="right",
        geometry_valid=True,
        surface_gap_m=-0.004,
        contact=True,
        collision=True,
        near=True,
        distance_gate=1.0,
    )
    result = EndEffectorSafetyResult(left=left, right=right)
    assert result.geometry_valid
    assert result.collision
    assert result.near
    assert np.isclose(result.min_surface_gap_m, -0.004)
    assert result.distance_gate == 1.0


def test_runtime_reads_exact_closest_surface_point(monkeypatch):
    class _AttachmentQuery:
        def get_closest_points(self, points, path):
            assert points == [(0.2, 0.3, 0.4)]
            assert path == "/World/Franka/panda_hand/collider"
            return {"dists": [0.01], "closest_points": [(0.1, 0.2, 0.3)]}

    monkeypatch.setitem(
        sys.modules,
        "carb",
        types.SimpleNamespace(Float3=lambda x, y, z: (x, y, z)),
    )
    runtime = PandaEndEffectorSafetyRuntime.__new__(PandaEndEffectorSafetyRuntime)
    runtime._attachment_query = _AttachmentQuery()
    runtime._closest_point_error_logged = set()
    result = HandSafetyResult(
        hand="left",
        geometry_valid=True,
        closest_collider_path="/World/Franka/panda_hand/collider",
    )

    point, valid = runtime.closest_surface_point_world_position(result, [0.2, 0.3, 0.4])

    assert valid
    assert np.allclose(point, [0.1, 0.2, 0.3])


def test_runtime_reads_direct_isaac_link_twist():
    class _Prim:
        def IsValid(self):
            return True

    class _Stage:
        def GetPrimAtPath(self, _path):
            return _Prim()

    class _RigidPrim:
        def get_linear_velocity(self):
            return np.array([0.1, 0.2, 0.3])

        def get_angular_velocity(self):
            return np.array([1.0, 2.0, 3.0])

    runtime = PandaEndEffectorSafetyRuntime.__new__(PandaEndEffectorSafetyRuntime)
    runtime.robot_prim_path = "/World/Franka"
    runtime._stage = _Stage()
    runtime._link_rigid_prims = {"panda_hand": _RigidPrim()}
    runtime._link_velocity_error_logged = set()
    result = HandSafetyResult(
        hand="left", geometry_valid=True, closest_link="panda_hand"
    )

    linear, angular, valid = runtime.closest_link_world_velocity(result)

    assert valid
    assert np.allclose(linear, [0.1, 0.2, 0.3])
    assert np.allclose(angular, [1.0, 2.0, 3.0])
