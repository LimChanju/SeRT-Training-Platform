from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_TASK_CHECKPOINT = (
    SCRIPT_DIR
    / "policies"
    / "ppo_pick_place_v7_residual_rewardv4_strict_best.pt"
)
SUPPORTED_ALGORITHMS = ("ppo", "sac", "td3")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate PPO/SAC/TD3 safety residuals on identical "
            "encounter manifests and environment-step budgets."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("train", "eval", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument(
        "--task-checkpoint",
        default=str(DEFAULT_TASK_CHECKPOINT),
    )
    parser.add_argument("--algorithms", default="ppo,sac,td3")
    parser.add_argument("--experiment-tag", default="encounter_benchmark_v1")
    parser.add_argument("--total-steps", type=int, default=30000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--eval-seeds", default="11,1011,2011")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--residual-alpha", type=float, default=0.1)
    parser.add_argument("--safety-gate-start-dist", type=float, default=0.13)
    parser.add_argument("--safety-gate-full-dist", type=float, default=0.05)
    parser.add_argument("--distance-progress-weight", type=float, default=2.0)
    parser.add_argument("--distance-progress-clip-m", type=float, default=0.03)
    parser.add_argument(
        "--encounter-severity-mix",
        default="safe=0.40,gate_only=0.25,near=0.20,near_miss=0.10,collision=0.05",
    )
    parser.add_argument(
        "--xyz-only-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    algorithms = _parse_algorithms(args.algorithms)
    train_manifest = Path(args.train_manifest).expanduser().resolve()
    eval_manifest = Path(args.eval_manifest).expanduser().resolve()
    task_checkpoint = Path(args.task_checkpoint).expanduser().resolve()
    for path in (train_manifest, eval_manifest, task_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    _validate_manifest_split(train_manifest, eval_manifest)
    eval_seeds = tuple(
        int(value.strip())
        for value in args.eval_seeds.split(",")
        if value.strip()
    )
    if not eval_seeds:
        raise ValueError("--eval-seeds must contain at least one seed.")
    tag = args.experiment_tag.strip()
    if not tag or any(character in tag for character in "/\\"):
        raise ValueError("--experiment-tag must be one directory name.")
    policy_dir = SCRIPT_DIR / "policies" / "encounter_benchmarks" / tag
    result_dir = SCRIPT_DIR / "eval_results" / "encounter_benchmarks" / tag

    if args.stage in ("train", "all"):
        _train(
            algorithms,
            train_manifest,
            task_checkpoint,
            policy_dir,
            args,
        )
    if args.stage in ("eval", "all"):
        _evaluate(
            algorithms,
            eval_manifest,
            task_checkpoint,
            policy_dir,
            result_dir,
            eval_seeds,
            args,
        )
    if args.stage in ("summarize", "all"):
        _summarize(algorithms, result_dir, eval_seeds)


def _train(
    algorithms: tuple[str, ...],
    train_manifest: Path,
    task_checkpoint: Path,
    policy_dir: Path,
    args: argparse.Namespace,
) -> None:
    policy_dir.mkdir(parents=True, exist_ok=True)
    for algorithm in algorithms:
        output = policy_dir / f"{algorithm}_safety.pt"
        best_output = policy_dir / f"{algorithm}_safety_best.pt"
        if not args.force and output.exists() and best_output.exists():
            print(f"[EncounterBenchmark] skip trained {algorithm}", flush=True)
            continue
        script = {
            "ppo": SCRIPT_DIR / "train_safety_residual.py",
            "sac": SCRIPT_DIR / "train_sac_safety_residual.py",
            "td3": SCRIPT_DIR / "train_td3_safety_residual.py",
        }[algorithm]
        command = _isaac_command(
            script,
            "--task-checkpoint",
            str(task_checkpoint),
            "--encounter-manifest",
            str(train_manifest),
            "--encounter-policy",
            "random",
            "--encounter-severity-mix",
            args.encounter_severity_mix,
            "--output",
            str(output),
            "--best-output",
            str(best_output),
            "--total-steps",
            str(args.total_steps),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--residual-alpha",
            str(args.residual_alpha),
            "--safety-gate-start-dist",
            str(args.safety_gate_start_dist),
            "--safety-gate-full-dist",
            str(args.safety_gate_full_dist),
            "--distance-progress-weight",
            str(args.distance_progress_weight),
            "--distance-progress-clip-m",
            str(args.distance_progress_clip_m),
            "--no-pseudo-errp",
        )
        if args.xyz_only_residual:
            command.append("--xyz-only-residual")
        if algorithm == "ppo":
            command.extend(
                (
                    "--rollout-steps",
                    str(args.rollout_steps),
                    "--gate-active-only",
                )
            )
        if algorithm == "sac" and args.xyz_only_residual:
            command.extend(("--target-entropy", "-3.0"))
        print(
            f"[EncounterBenchmark] train algorithm={algorithm} "
            f"steps={args.total_steps}",
            flush=True,
        )
        _run(command)


def _evaluate(
    algorithms: tuple[str, ...],
    eval_manifest: Path,
    task_checkpoint: Path,
    policy_dir: Path,
    result_dir: Path,
    eval_seeds: tuple[int, ...],
    args: argparse.Namespace,
) -> None:
    models = ("task_only", *algorithms)
    for model in models:
        safety_checkpoint = (
            None
            if model == "task_only"
            else policy_dir / f"{model}_safety_best.pt"
        )
        if safety_checkpoint is not None and not safety_checkpoint.exists():
            raise FileNotFoundError(safety_checkpoint)
        for seed in eval_seeds:
            output_dir = result_dir / model
            output_json = output_dir / f"seed_{seed}.json"
            output_csv = output_dir / f"seed_{seed}.csv"
            output_steps = output_dir / f"seed_{seed}_steps.csv"
            if (
                not args.force
                and output_json.exists()
                and output_csv.exists()
                and output_steps.exists()
            ):
                print(
                    f"[EncounterBenchmark] skip eval model={model} seed={seed}",
                    flush=True,
                )
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            command = _isaac_command(
                SCRIPT_DIR / "evaluate_rollout_policy.py",
                "--checkpoint",
                str(task_checkpoint),
                "--encounter-manifest",
                str(eval_manifest),
                "--encounter-policy",
                "cycle",
                "--episodes",
                "0",
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--mask-human-obs-for-policy",
                "--safety-gate-start-dist",
                str(args.safety_gate_start_dist),
                "--safety-gate-full-dist",
                str(args.safety_gate_full_dist),
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
                "--output-step-csv",
                str(output_steps),
                "--no-pseudo-errp",
            )
            if safety_checkpoint is not None:
                command.extend(
                    (
                        "--safety-residual-checkpoint",
                        str(safety_checkpoint),
                        "--safety-residual-alpha",
                        str(args.residual_alpha),
                    )
                )
            print(
                f"[EncounterBenchmark] eval model={model} seed={seed}",
                flush=True,
            )
            _run(command)


def _summarize(
    algorithms: tuple[str, ...],
    result_dir: Path,
    eval_seeds: tuple[int, ...],
) -> None:
    rows: list[dict[str, Any]] = []
    for model in ("task_only", *algorithms):
        for seed in eval_seeds:
            path = result_dir / model / f"seed_{seed}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for episode in payload["episodes"]:
                rows.append(
                    {
                        "model": model,
                        "eval_seed": seed,
                        "episode": int(episode["episode"]),
                        "encounter_id": episode.get("encounter_id", ""),
                        "target_severity": episode.get(
                            "encounter_target_severity",
                            "",
                        ),
                        "realized_severity": episode.get(
                            "encounter_realized_severity",
                            "",
                        ),
                        "success": int(bool(episode["success"])),
                        "steps": int(episode["steps"]),
                        "total_reward": float(episode["total_reward"]),
                        "collision_rate": float(
                            episode.get("collision_rate", 0.0)
                        ),
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
                        "mean_residual_norm": float(
                            episode.get("mean_safety_residual_norm", 0.0)
                        ),
                        "mean_applied_position_m": float(
                            episode.get(
                                "mean_safety_applied_position_m",
                                0.0,
                            )
                        ),
                    }
                )
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(result_dir / "episode_results.csv", rows)
    metrics = (
        "success",
        "steps",
        "total_reward",
        "collision_rate",
        "near_miss_rate",
        "near_rate",
        "gate_activation_rate",
        "min_surface_gap_m",
        "mean_residual_norm",
        "mean_applied_position_m",
    )
    summary: dict[str, Any] = {
        "pairing_unit": "eval_seed + encounter_id",
        "models": {},
    }
    for model in ("task_only", *algorithms):
        selected = [row for row in rows if row["model"] == model]
        summary["models"][model] = {
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in metrics
        }
        summary["models"][model]["episodes"] = len(selected)
        summary["models"][model]["by_target_severity"] = {}
        for severity in (
            "safe",
            "gate_only",
            "near",
            "near_miss",
            "collision",
        ):
            subset = [
                row
                for row in selected
                if row["target_severity"] == severity
            ]
            if not subset:
                continue
            summary["models"][model]["by_target_severity"][severity] = {
                metric: float(np.mean([row[metric] for row in subset]))
                for metric in metrics
            }
            summary["models"][model]["by_target_severity"][severity][
                "episodes"
            ] = len(subset)
    with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[EncounterBenchmark] saved summaries under {result_dir}", flush=True)


def _parse_algorithms(value: str) -> tuple[str, ...]:
    algorithms = tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    unknown = sorted(set(algorithms) - set(SUPPORTED_ALGORITHMS))
    if not algorithms or unknown:
        raise ValueError(
            f"Unknown algorithms {unknown}; supported={SUPPORTED_ALGORITHMS}"
        )
    return algorithms


def _validate_manifest_split(
    train_manifest: Path,
    eval_manifest: Path,
) -> None:
    manifests = []
    for path in (train_manifest, eval_manifest):
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != "hri_encounter_manifest_v2":
            raise ValueError(f"Unsupported encounter manifest: {path}")
        if not manifest.get("scenarios"):
            raise ValueError(f"Encounter manifest has no scenarios: {path}")
        manifests.append(manifest)
    train_sessions = {
        str(item.get("session_id", ""))
        for item in manifests[0]["scenarios"]
    }
    eval_sessions = {
        str(item.get("session_id", ""))
        for item in manifests[1]["scenarios"]
    }
    overlap = sorted((train_sessions & eval_sessions) - {""})
    if overlap:
        raise ValueError(
            "Train/eval encounter manifests share collection sessions: "
            f"{overlap}"
        )
    print(
        f"[EncounterBenchmark] train_scenarios="
        f"{len(manifests[0]['scenarios'])} "
        f"eval_scenarios={len(manifests[1]['scenarios'])} "
        f"train_sessions={len(train_sessions)} "
        f"eval_sessions={len(eval_sessions)}",
        flush=True,
    )


def _isaac_command(script: Path, *arguments: str) -> list[str]:
    return [
        str(PROJECT_DIR / "launch_isaac.sh"),
        str(script),
        *arguments,
    ]


def _run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["ISAAC_SKIP_VR_WAIT"] = "1"
    subprocess.run(
        command,
        cwd=PROJECT_DIR,
        env=environment,
        check=True,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
