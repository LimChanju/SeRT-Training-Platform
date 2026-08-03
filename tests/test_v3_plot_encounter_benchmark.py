import pytest

from v3_chan.plot_encounter_benchmark import _paired_comparisons


def _row(
    model: str,
    encounter_id: str,
    layout_id: str,
    *,
    success: int,
    collision_rate: float,
):
    return {
        "model": model,
        "eval_seed": "11",
        "encounter_id": encounter_id,
        "scene_layout_id": layout_id,
        "success": str(success),
        "collision_rate": str(collision_rate),
        "near_rate": "0.2",
        "near_miss_rate": "0.05",
        "min_surface_gap_m": "0.01",
        "steps": "620",
        "total_reward": "-100",
    }


def test_paired_comparison_uses_matching_seed_encounter_and_layout():
    rows = [
        _row("task_only", "a", "layout-a", success=0, collision_rate=0.1),
        _row("task_only", "b", "layout-b", success=1, collision_rate=0.0),
        _row("ppo", "a", "layout-a", success=1, collision_rate=0.0),
        _row("ppo", "b", "layout-b", success=1, collision_rate=0.0),
    ]

    result = _paired_comparisons(rows, ("ppo",))["ppo"]

    assert result["pairs"] == 2
    assert result["layout_pairing_verified"] is True
    assert result["metrics"]["success_pp"]["paired_difference"] == 50.0
    assert result["metrics"]["collision_rate_pp"]["paired_difference"] == -5.0


def test_paired_comparison_rejects_layout_mismatch():
    rows = [
        _row("task_only", "a", "layout-a", success=1, collision_rate=0.0),
        _row("ppo", "a", "different", success=1, collision_rate=0.0),
    ]

    with pytest.raises(ValueError, match="Scene layout mismatch"):
        _paired_comparisons(rows, ("ppo",))
