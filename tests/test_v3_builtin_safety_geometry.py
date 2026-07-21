import math

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
