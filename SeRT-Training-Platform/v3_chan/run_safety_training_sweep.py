from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
TASK_CHECKPOINT = (
    SCRIPT_DIR / "policies" / "ppo_pick_place_v7_residual_rewardv4_strict_best.pt"
)
CONFIGS = {
    "conservative": {
        "distance_progress_weight": 5.0,
        "lr": 3e-5,
        "clip_ratio": 0.05,
        "update_epochs": 2,
        "entropy_coef": 0.0,
    },
    "balanced": {
        "distance_progress_weight": 10.0,
        "lr": 1e-4,
        "clip_ratio": 0.1,
        "update_epochs": 2,
        "entropy_coef": 0.001,
    },
    "strong_reward": {
        "distance_progress_weight": 20.0,
        "lr": 1e-4,
        "clip_ratio": 0.1,
        "update_epochs": 2,
        "entropy_coef": 0.001,
    },
    "strong_update": {
        "distance_progress_weight": 10.0,
        "lr": 1e-4,
        "clip_ratio": 0.1,
        "update_epochs": 4,
        "entropy_coef": 0.001,
    },
    "direction_010": {
        "distance_progress_weight": 10.0,
        "avoidance_direction_weight": 0.1,
        "avoidance_target_residual_norm": 0.01,
        "lr": 1e-4,
        "clip_ratio": 0.1,
        "update_epochs": 2,
        "entropy_coef": 0.001,
    },
    "direction_025": {
        "distance_progress_weight": 10.0,
        "avoidance_direction_weight": 0.25,
        "avoidance_target_residual_norm": 0.01,
        "lr": 1e-4,
        "clip_ratio": 0.1,
        "update_epochs": 2,
        "entropy_coef": 0.001,
    },
    "direction_050": {
        "distance_progress_weight": 10.0,
        "avoidance_direction_weight": 0.5,
        "avoidance_target_residual_norm": 0.01,
        "lr": 1e-4,
        "clip_ratio": 0.1,
        "update_epochs": 2,
        "entropy_coef": 0.001,
    },
    "aux_001": {
        "distance_progress_weight": 10.0,
        "avoidance_aux_coef": 1.0,
        "avoidance_aux_target_norm": 0.01,
        "lr": 1e-4,
        "clip_ratio": 0.1,
        "update_epochs": 2,
        "entropy_coef": 0.001,
    },
    "aux_010": {
        "distance_progress_weight": 10.0,
        "avoidance_aux_coef": 10.0,
        "avoidance_aux_target_norm": 0.01,
        "lr": 1e-4,
        "clip_ratio": 0.1,
        "update_epochs": 2,
        "entropy_coef": 0.001,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen stronger physical PPO safety residual settings."
    )
    parser.add_argument(
        "--stage", choices=("train", "eval", "summarize", "all"), default="all"
    )
    parser.add_argument("--experiment-tag", default="surface_gap_new_sessions_v1")
    parser.add_argument("--sweep-tag", default="strong_residual_screen_v1")
    parser.add_argument("--folds", default="1")
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--total-steps", type=int, default=30000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--eval-seeds", default="11,1011,2011")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--residual-alpha", type=float, default=0.3)
    parser.add_argument("--safety-gate-start-dist", type=float, default=0.13)
    parser.add_argument("--safety-gate-full-dist", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    folds = tuple(
        int(value.strip()) for value in args.folds.split(",") if value.strip()
    )
    if not folds or any(fold < 1 or fold > 4 for fold in folds):
        raise ValueError("--folds must contain values from 1 to 4")
    configs = tuple(value.strip() for value in args.configs.split(",") if value.strip())
    unknown = sorted(set(configs) - set(CONFIGS))
    if not configs or unknown:
        raise ValueError(
            f"Unknown configs: {unknown}; expected one of {tuple(CONFIGS)}"
        )
    seeds = tuple(
        int(value.strip()) for value in args.eval_seeds.split(",") if value.strip()
    )
    if not seeds:
        raise ValueError("--eval-seeds must contain at least one seed")

    cv_dir = SCRIPT_DIR / "trajectories" / "cv4" / args.experiment_tag
    policy_dir = SCRIPT_DIR / "policies" / "safety_sweeps" / args.sweep_tag
    result_dir = SCRIPT_DIR / "eval_results" / "safety_sweeps" / args.sweep_tag
    required = [TASK_CHECKPOINT]
    for fold in folds:
        required.extend(
            (
                cv_dir / f"fold_{fold:02d}" / "train.hdf5",
                cv_dir / f"fold_{fold:02d}" / "eval.hdf5",
            )
        )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    if args.stage in ("train", "all"):
        _train(args, folds, configs, cv_dir, policy_dir)
    if args.stage in ("eval", "all"):
        _evaluate(args, folds, configs, seeds, cv_dir, policy_dir, result_dir)
    if args.stage in ("summarize", "all"):
        _summarize(args, folds, configs, seeds, result_dir)


def _train(
    args: argparse.Namespace,
    folds: tuple[int, ...],
    configs: tuple[str, ...],
    cv_dir: Path,
    policy_dir: Path,
) -> None:
    for fold in folds:
        train_data = cv_dir / f"fold_{fold:02d}" / "train.hdf5"
        for name in configs:
            config = CONFIGS[name]
            output = policy_dir / name / f"fold_{fold:02d}" / "ppo_safety_physical.pt"
            best_output = output.with_name("ppo_safety_physical_best.pt")
            if not args.force and output.exists() and best_output.exists():
                print(
                    f"[TrainingSweep] skip trained fold={fold} config={name}",
                    flush=True,
                )
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            command = _isaac_command(
                SCRIPT_DIR / "train_safety_residual.py",
                "--task-checkpoint",
                str(TASK_CHECKPOINT),
                "--human-replay-data",
                str(train_data),
                "--output",
                str(output),
                "--best-output",
                str(best_output),
                "--total-steps",
                str(args.total_steps),
                "--rollout-steps",
                str(args.rollout_steps),
                "--device",
                args.device,
                "--residual-alpha",
                str(args.residual_alpha),
                "--safety-gate-start-dist",
                str(args.safety_gate_start_dist),
                "--safety-gate-full-dist",
                str(args.safety_gate_full_dist),
                "--distance-progress-weight",
                str(config["distance_progress_weight"]),
                "--lr",
                str(config["lr"]),
                "--clip-ratio",
                str(config["clip_ratio"]),
                "--update-epochs",
                str(config["update_epochs"]),
                "--entropy-coef",
                str(config["entropy_coef"]),
                "--seed",
                str(23 + fold),
                "--gate-active-only",
                "--xyz-only-residual",
                "--no-pseudo-errp",
            )
            if float(config.get("avoidance_direction_weight", 0.0)) != 0.0:
                command.extend(
                    (
                        "--avoidance-direction-weight",
                        str(config["avoidance_direction_weight"]),
                        "--avoidance-target-residual-norm",
                        str(config["avoidance_target_residual_norm"]),
                    )
                )
            if float(config.get("avoidance_aux_coef", 0.0)) != 0.0:
                command.extend(
                    (
                        "--avoidance-aux-coef",
                        str(config["avoidance_aux_coef"]),
                        "--avoidance-aux-target-norm",
                        str(config["avoidance_aux_target_norm"]),
                    )
                )
            print(
                f"[TrainingSweep] train fold={fold} config={name} settings={config}",
                flush=True,
            )
            _run(command)
            if not output.exists() or not best_output.exists():
                raise RuntimeError(
                    f"Training did not write checkpoints for fold={fold} {name}"
                )


def _evaluate(
    args: argparse.Namespace,
    folds: tuple[int, ...],
    configs: tuple[str, ...],
    seeds: tuple[int, ...],
    cv_dir: Path,
    policy_dir: Path,
    result_dir: Path,
) -> None:
    for fold in folds:
        eval_data = cv_dir / f"fold_{fold:02d}" / "eval.hdf5"
        with h5py.File(eval_data, "r") as f:
            episode_count = len(f["episodes"])
        for name in configs:
            checkpoint = (
                policy_dir / name / f"fold_{fold:02d}" / "ppo_safety_physical_best.pt"
            )
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            for seed in seeds:
                output_dir = result_dir / name / f"fold_{fold:02d}"
                output_json = output_dir / f"seed_{seed}.json"
                output_csv = output_dir / f"seed_{seed}.csv"
                if not args.force and output_json.exists() and output_csv.exists():
                    print(
                        f"[TrainingSweep] skip evaluated fold={fold} config={name} seed={seed}",
                        flush=True,
                    )
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)
                command = _isaac_command(
                    SCRIPT_DIR / "evaluate_rollout_policy.py",
                    "--checkpoint",
                    str(TASK_CHECKPOINT),
                    "--human-replay-data",
                    str(eval_data),
                    "--episodes",
                    str(episode_count),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--mask-human-obs-for-policy",
                    "--safety-gate-start-dist",
                    str(args.safety_gate_start_dist),
                    "--safety-gate-full-dist",
                    str(args.safety_gate_full_dist),
                    "--safety-residual-checkpoint",
                    str(checkpoint),
                    "--safety-residual-alpha",
                    str(args.residual_alpha),
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(output_csv),
                )
                print(
                    f"[TrainingSweep] eval fold={fold} config={name} seed={seed}",
                    flush=True,
                )
                _run(command)
                if not output_json.exists() or not output_csv.exists():
                    raise RuntimeError(
                        f"Evaluation did not write outputs for fold={fold} {name} seed={seed}"
                    )


def _summarize(
    args: argparse.Namespace,
    folds: tuple[int, ...],
    configs: tuple[str, ...],
    seeds: tuple[int, ...],
    result_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for name in configs:
            for seed in seeds:
                path = result_dir / name / f"fold_{fold:02d}" / f"seed_{seed}.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                for episode in payload["episodes"]:
                    steps = max(1, int(episode["steps"]))
                    rows.append(
                        {
                            "fold": fold,
                            "config": name,
                            "eval_seed": seed,
                            "episode": int(episode["episode"]),
                            "success": int(bool(episode["success"])),
                            "steps": steps,
                            "collision_rate": float(
                                episode.get("human_collision_count", 0)
                            )
                            / steps,
                            "near_rate": float(episode.get("near_human_count", 0))
                            / steps,
                            "near_miss_rate": float(episode.get("near_miss_count", 0))
                            / steps,
                            "min_surface_gap_m": float(
                                episode.get("min_hand_gripper_surface_gap", 10.0)
                            ),
                            "residual_norm": float(
                                episode.get("mean_safety_residual_norm", 0.0)
                            ),
                            "applied_position_m": float(
                                episode.get("mean_safety_applied_position_m", 0.0)
                            ),
                            "max_applied_position_m": float(
                                episode.get("max_safety_applied_position_m", 0.0)
                            ),
                        }
                    )
    summary = []
    for name in configs:
        fold_records = []
        for fold in folds:
            selected = [
                row for row in rows if row["config"] == name and row["fold"] == fold
            ]
            record: dict[str, Any] = {"fold": fold}
            for metric in (
                "success",
                "steps",
                "collision_rate",
                "near_rate",
                "near_miss_rate",
                "min_surface_gap_m",
                "residual_norm",
                "applied_position_m",
                "max_applied_position_m",
            ):
                record[metric] = float(
                    np.mean([float(row[metric]) for row in selected])
                )
            fold_records.append(record)
        record = {
            "config": name,
            **CONFIGS[name],
            "rollouts": sum(1 for row in rows if row["config"] == name),
        }
        for metric in (
            "success",
            "steps",
            "collision_rate",
            "near_rate",
            "near_miss_rate",
            "min_surface_gap_m",
            "residual_norm",
            "applied_position_m",
            "max_applied_position_m",
        ):
            record[metric] = float(
                np.mean([float(row[metric]) for row in fold_records])
            )
        summary.append(record)
    eligible = [row for row in summary if float(row["success"]) >= 0.95]
    eligible.sort(
        key=lambda row: (
            float(row["collision_rate"]),
            -float(row["min_surface_gap_m"]),
            float(row["near_rate"]),
        )
    )
    for rank, row in enumerate(eligible, start=1):
        row["rank"] = rank
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(result_dir / "episode_results.csv", rows)
    _write_csv(result_dir / "ranking.csv", eligible)
    payload = {
        "folds": list(folds),
        "residual_alpha": args.residual_alpha,
        "selection_rule": "success >= 0.95, then collision, minimum surface gap, near rate",
        "ranking": eligible,
    }
    with (result_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"[TrainingSweep] saved summary under {result_dir}", flush=True)


def _isaac_command(script: Path, *arguments: str) -> list[str]:
    return [str(PROJECT_DIR / "launch_isaac.sh"), str(script), *arguments]


def _run(command: list[str]) -> None:
    env = os.environ.copy()
    env["ISAAC_SKIP_VR_WAIT"] = "1"
    subprocess.run(command, cwd=PROJECT_DIR, env=env, check=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
