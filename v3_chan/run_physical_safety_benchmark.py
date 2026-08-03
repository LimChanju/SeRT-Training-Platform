from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from physical_safety_controllers import PHYSICAL_SAFETY_MODES


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_TASK_CHECKPOINT = (
    SCRIPT_DIR
    / "policies"
    / "ppo_pick_place_v7_residual_rewardv4_strict_best.pt"
)
DEFAULT_MODES = ("none", "rmpflow", "cbf", "rmpflow_cbf")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate analytic physical-safety controllers on paired held-out "
            "encounters. The frozen task policy and scene seeds are identical "
            "for every controller."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("eval", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--task-checkpoint", default=str(DEFAULT_TASK_CHECKPOINT))
    parser.add_argument("--controllers", default=",".join(DEFAULT_MODES))
    parser.add_argument("--eval-seeds", default="11,1011,2011")
    parser.add_argument("--experiment-tag", default="physical_safety_v1")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--eval-log-every", type=int, default=25)
    parser.add_argument(
        "--encounter-timebase",
        choices=("recorded", "step"),
        default="recorded",
    )
    parser.add_argument("--encounter-playback-speed", type=float, default=1.0)
    parser.add_argument("--rmpflow-human-safety-margin-m", type=float, default=0.05)
    parser.add_argument("--cbf-safe-gap-m", type=float, default=0.05)
    parser.add_argument("--cbf-activation-gap-m", type=float, default=0.13)
    parser.add_argument("--cbf-gamma-per-s", type=float, default=8.0)
    parser.add_argument("--cbf-prediction-horizon-s", type=float, default=0.15)
    parser.add_argument("--cbf-max-prediction-buffer-m", type=float, default=0.08)
    parser.add_argument("--cbf-max-joint-speed-rad-s", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    controllers = _parse_controllers(args.controllers)
    seeds = _parse_seeds(args.eval_seeds)
    manifest = Path(args.eval_manifest).expanduser().resolve()
    task_checkpoint = Path(args.task_checkpoint).expanduser().resolve()
    for path in (manifest, task_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    expected_episodes = _manifest_episode_count(manifest)
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if (
        not np.isfinite(args.encounter_playback_speed)
        or args.encounter_playback_speed <= 0.0
    ):
        raise ValueError("--encounter-playback-speed must be finite and positive")
    tag = args.experiment_tag.strip()
    if not tag or any(character in tag for character in "/\\"):
        raise ValueError("--experiment-tag must be one directory name")
    result_dir = SCRIPT_DIR / "eval_results" / "physical_safety" / tag

    if args.stage in ("eval", "all"):
        _evaluate(
            controllers=controllers,
            seeds=seeds,
            manifest=manifest,
            task_checkpoint=task_checkpoint,
            result_dir=result_dir,
            expected_episodes=expected_episodes,
            args=args,
        )
    if args.stage in ("summarize", "all"):
        _summarize(controllers, seeds, result_dir)


def _evaluate(
    *,
    controllers: tuple[str, ...],
    seeds: tuple[int, ...],
    manifest: Path,
    task_checkpoint: Path,
    result_dir: Path,
    expected_episodes: int,
    args: argparse.Namespace,
) -> None:
    for controller in controllers:
        for seed in seeds:
            output_dir = result_dir / controller
            output_json = output_dir / f"seed_{seed}.json"
            output_csv = output_dir / f"seed_{seed}.csv"
            output_steps = output_dir / f"seed_{seed}_steps.csv"
            outputs = (output_json, output_csv, output_steps)
            if not args.force and all(path.exists() for path in outputs):
                _validate_result(output_json, expected_episodes)
                print(
                    f"[PhysicalSafetyBenchmark] skip controller={controller} "
                    f"seed={seed}",
                    flush=True,
                )
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            if args.force:
                for path in outputs:
                    path.unlink(missing_ok=True)
            command = [
                str(PROJECT_DIR / "launch_isaac.sh"),
                str(SCRIPT_DIR / "evaluate_rollout_policy.py"),
                "--checkpoint",
                str(task_checkpoint),
                "--encounter-manifest",
                str(manifest),
                "--encounter-policy",
                "cycle",
                "--encounter-timebase",
                args.encounter_timebase,
                "--encounter-playback-speed",
                str(args.encounter_playback_speed),
                "--episodes",
                "0",
                "--max-steps",
                str(args.max_steps),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--mask-human-obs-for-policy",
                "--no-pseudo-errp",
                "--physical-safety-controller",
                controller,
                "--rmpflow-human-safety-margin-m",
                str(args.rmpflow_human_safety_margin_m),
                "--cbf-safe-gap-m",
                str(args.cbf_safe_gap_m),
                "--cbf-activation-gap-m",
                str(args.cbf_activation_gap_m),
                "--cbf-gamma-per-s",
                str(args.cbf_gamma_per_s),
                "--cbf-prediction-horizon-s",
                str(args.cbf_prediction_horizon_s),
                "--cbf-max-prediction-buffer-m",
                str(args.cbf_max_prediction_buffer_m),
                "--cbf-max-joint-speed-rad-s",
                str(args.cbf_max_joint_speed_rad_s),
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
                "--output-step-csv",
                str(output_steps),
                "--log-every",
                str(args.eval_log_every),
            ]
            print(
                f"[PhysicalSafetyBenchmark] eval controller={controller} "
                f"seed={seed} episodes={expected_episodes}",
                flush=True,
            )
            environment = os.environ.copy()
            environment["ISAAC_SKIP_VR_WAIT"] = "1"
            subprocess.run(
                command,
                cwd=PROJECT_DIR,
                env=environment,
                check=True,
            )
            for path in outputs:
                if not path.exists():
                    raise RuntimeError(f"Evaluation did not produce {path}")
            _validate_result(output_json, expected_episodes)


def _summarize(
    controllers: tuple[str, ...],
    seeds: tuple[int, ...],
    result_dir: Path,
) -> None:
    episode_rows: list[dict[str, Any]] = []
    for controller in controllers:
        for seed in seeds:
            path = result_dir / controller / f"seed_{seed}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for episode in payload.get("episodes", ()):
                episode_rows.append(
                    {
                        "controller": controller,
                        "eval_seed": seed,
                        "episode": int(episode["episode"]),
                        "encounter_id": str(episode.get("encounter_id", "")),
                        "scene_layout_id": str(episode.get("scene_layout_id", "")),
                        "target_severity": str(
                            episode.get("encounter_target_severity", "")
                        ),
                        "success": int(bool(episode["success"])),
                        "steps": int(episode["steps"]),
                        "total_reward": float(episode["total_reward"]),
                        "collision_rate": float(episode.get("collision_rate", 0.0)),
                        "near_miss_rate": float(
                            episode.get("near_miss_rate", 0.0)
                        ),
                        "near_rate": float(episode.get("near_rate", 0.0)),
                        "gate_activation_rate": float(
                            episode.get("gate_activation_rate", 0.0)
                        ),
                        "min_surface_gap_m": float(
                            episode.get("min_surface_gap", 10.0)
                        ),
                        "physical_active_rate": float(
                            episode.get("physical_safety_active_rate", 0.0)
                        ),
                        "physical_intervention_rate": float(
                            episode.get("physical_safety_intervention_rate", 0.0)
                        ),
                        "mean_intervention_norm_radps": float(
                            episode.get(
                                "mean_physical_safety_intervention_norm_radps",
                                0.0,
                            )
                        ),
                        "max_intervention_norm_radps": float(
                            episode.get(
                                "max_physical_safety_intervention_norm_radps",
                                0.0,
                            )
                        ),
                        "feasible_rate": float(
                            episode.get("physical_safety_feasible_rate", 1.0)
                        ),
                        "mean_slack_radps": float(
                            episode.get("mean_physical_safety_slack_radps", 0.0)
                        ),
                        "mean_solve_time_ms": float(
                            episode.get(
                                "mean_physical_safety_solve_time_ms",
                                0.0,
                            )
                        ),
                        "ee_path_length_m": float(
                            episode.get("ee_path_length_m", 0.0)
                        ),
                        "rms_ee_acceleration_mps2": float(
                            episode.get("rms_ee_acceleration_mps2", 0.0)
                        ),
                        "p95_ee_jerk_mps3": float(
                            episode.get("p95_ee_jerk_mps3", 0.0)
                        ),
                        "rms_ee_jerk_mps3": float(
                            episode.get("rms_ee_jerk_mps3", 0.0)
                        ),
                        "max_ee_jerk_mps3": float(
                            episode.get("max_ee_jerk_mps3", 0.0)
                        ),
                        "integrated_squared_ee_jerk_m2ps5": float(
                            episode.get(
                                "integrated_squared_ee_jerk_m2ps5",
                                0.0,
                            )
                        ),
                        "rms_gate_ee_acceleration_mps2": float(
                            episode.get("rms_gate_ee_acceleration_mps2", 0.0)
                        ),
                        "p95_gate_ee_jerk_mps3": float(
                            episode.get("p95_gate_ee_jerk_mps3", 0.0)
                        ),
                        "rms_gate_ee_jerk_mps3": float(
                            episode.get("rms_gate_ee_jerk_mps3", 0.0)
                        ),
                        "max_gate_ee_jerk_mps3": float(
                            episode.get("max_gate_ee_jerk_mps3", 0.0)
                        ),
                    }
                )
    _validate_pairing(episode_rows, controllers)
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(result_dir / "episode_results.csv", episode_rows)

    metrics = (
        "success",
        "steps",
        "total_reward",
        "collision_rate",
        "near_miss_rate",
        "near_rate",
        "gate_activation_rate",
        "min_surface_gap_m",
        "physical_active_rate",
        "physical_intervention_rate",
        "mean_intervention_norm_radps",
        "max_intervention_norm_radps",
        "feasible_rate",
        "mean_slack_radps",
        "mean_solve_time_ms",
        "ee_path_length_m",
        "rms_ee_acceleration_mps2",
        "p95_ee_jerk_mps3",
        "rms_ee_jerk_mps3",
        "max_ee_jerk_mps3",
        "integrated_squared_ee_jerk_m2ps5",
        "rms_gate_ee_acceleration_mps2",
        "p95_gate_ee_jerk_mps3",
        "rms_gate_ee_jerk_mps3",
        "max_gate_ee_jerk_mps3",
    )
    summary: dict[str, Any] = {
        "pairing_unit": "eval_seed + encounter_id + scene_layout_id",
        "reference_controller": "none" if "none" in controllers else "",
        "controllers": {},
    }
    for controller in controllers:
        selected = [
            row for row in episode_rows if row["controller"] == controller
        ]
        summary["controllers"][controller] = {
            "episodes": len(selected),
            **{
                metric: float(np.mean([row[metric] for row in selected]))
                for metric in metrics
            },
        }
    if "none" in controllers:
        reference = {
            _pairing_key(row): row
            for row in episode_rows
            if row["controller"] == "none"
        }
        for controller in controllers:
            if controller == "none":
                continue
            selected = [
                row for row in episode_rows if row["controller"] == controller
            ]
            summary["controllers"][controller]["paired_delta_vs_none"] = {
                metric: float(
                    np.mean(
                        [
                            row[metric] - reference[_pairing_key(row)][metric]
                            for row in selected
                        ]
                    )
                )
                for metric in metrics
            }
    with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[PhysicalSafetyBenchmark] saved {result_dir}", flush=True)


def _parse_controllers(value: str) -> tuple[str, ...]:
    controllers = tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    unknown = sorted(set(controllers) - set(PHYSICAL_SAFETY_MODES))
    if not controllers or unknown:
        raise ValueError(
            f"Unknown physical safety controllers {unknown}; "
            f"supported={PHYSICAL_SAFETY_MODES}"
        )
    return controllers


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("--eval-seeds must contain at least one seed")
    return seeds


def _manifest_episode_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    scenarios = payload.get("scenarios", ())
    if not scenarios:
        raise ValueError(f"Encounter manifest has no scenarios: {path}")
    return len(scenarios)


def _validate_result(path: Path, expected_episodes: int) -> None:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    actual = len(payload.get("episodes", ()))
    if actual != expected_episodes:
        raise RuntimeError(
            f"Incomplete evaluation {path}: episodes={actual}, "
            f"expected={expected_episodes}"
        )


def _pairing_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["eval_seed"]),
        str(row["encounter_id"]),
        str(row["scene_layout_id"]),
    )


def _validate_pairing(
    rows: list[dict[str, Any]], controllers: tuple[str, ...]
) -> None:
    expected = None
    for controller in controllers:
        keys = {
            _pairing_key(row) for row in rows if row["controller"] == controller
        }
        if expected is None:
            expected = keys
        elif keys != expected:
            raise RuntimeError(
                f"Paired evaluation mismatch for controller={controller}: "
                f"expected={len(expected)} actual={len(keys)}"
            )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
