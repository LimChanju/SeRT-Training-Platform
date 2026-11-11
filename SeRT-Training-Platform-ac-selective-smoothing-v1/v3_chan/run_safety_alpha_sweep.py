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
DEFAULT_EXPERIMENT = "surface_gap_new_sessions_v1"
TASK_CHECKPOINT = (
    SCRIPT_DIR / "policies" / "ppo_pick_place_v7_residual_rewardv4_strict_best.pt"
)
METRICS = (
    "success",
    "steps",
    "collision_rate",
    "near_rate",
    "near_miss_rate",
    "gate_rate",
    "min_surface_gap_m",
    "residual_norm",
    "applied_position_m",
    "max_applied_position_m",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained safety policies over a residual-alpha sweep."
    )
    parser.add_argument("--stage", choices=("eval", "summarize", "all"), default="all")
    parser.add_argument("--experiment-tag", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--folds", default="1,2,3,4")
    parser.add_argument("--variants", default="physical,proxy")
    parser.add_argument("--alphas", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--eval-seeds", default="11,1011,2011")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--safety-gate-start-dist", type=float, default=0.13)
    parser.add_argument("--safety-gate-full-dist", type=float, default=0.05)
    parser.add_argument(
        "--reuse-base-alpha",
        type=float,
        default=0.1,
        help="Reuse the existing CV evaluation for this alpha instead of launching Isaac again.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    folds = _parse_ints(args.folds)
    seeds = _parse_ints(args.eval_seeds)
    variants = tuple(
        value.strip() for value in args.variants.split(",") if value.strip()
    )
    alphas = tuple(
        float(value.strip()) for value in args.alphas.split(",") if value.strip()
    )
    if not variants or any(value not in ("physical", "proxy") for value in variants):
        raise ValueError("--variants must contain physical and/or proxy")
    if not alphas or any(value <= 0.0 for value in alphas):
        raise ValueError("--alphas must contain positive values")

    cv_dir = SCRIPT_DIR / "trajectories" / "cv4" / args.experiment_tag
    policy_dir = SCRIPT_DIR / "policies" / "cv4" / args.experiment_tag
    base_result_dir = SCRIPT_DIR / "eval_results" / "cv4" / args.experiment_tag
    sweep_dir = base_result_dir / "alpha_sweep"
    _validate_inputs(
        cv_dir,
        policy_dir,
        base_result_dir,
        folds,
        variants,
        seeds,
        args.reuse_base_alpha,
    )

    if args.stage in ("eval", "all"):
        _evaluate(args, cv_dir, policy_dir, sweep_dir, folds, variants, alphas, seeds)
    if args.stage in ("summarize", "all"):
        _summarize(
            base_result_dir,
            sweep_dir,
            folds,
            variants,
            alphas,
            seeds,
            args.reuse_base_alpha,
        )


def _evaluate(
    args: argparse.Namespace,
    cv_dir: Path,
    policy_dir: Path,
    sweep_dir: Path,
    folds: tuple[int, ...],
    variants: tuple[str, ...],
    alphas: tuple[float, ...],
    seeds: tuple[int, ...],
) -> None:
    for alpha in alphas:
        if np.isclose(alpha, args.reuse_base_alpha):
            print(f"[AlphaSweep] reuse existing alpha={alpha:g} evaluation", flush=True)
            continue
        alpha_tag = _alpha_tag(alpha)
        for fold in folds:
            eval_data = cv_dir / f"fold_{fold:02d}" / "eval.hdf5"
            with h5py.File(eval_data, "r") as f:
                episode_count = len(f["episodes"])
            for variant in variants:
                checkpoint = (
                    policy_dir / f"fold_{fold:02d}" / f"ppo_safety_{variant}_best.pt"
                )
                for seed in seeds:
                    output_dir = sweep_dir / alpha_tag / f"fold_{fold:02d}"
                    output_json = output_dir / f"{variant}_seed_{seed}.json"
                    output_csv = output_dir / f"{variant}_seed_{seed}.csv"
                    if not args.force and output_json.exists() and output_csv.exists():
                        print(
                            f"[AlphaSweep] skip alpha={alpha:g} fold={fold} "
                            f"variant={variant} seed={seed}",
                            flush=True,
                        )
                        continue
                    output_dir.mkdir(parents=True, exist_ok=True)
                    command = [
                        str(PROJECT_DIR / "launch_isaac.sh"),
                        str(SCRIPT_DIR / "evaluate_rollout_policy.py"),
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
                        str(alpha),
                        "--output-json",
                        str(output_json),
                        "--output-csv",
                        str(output_csv),
                    ]
                    print(
                        f"[AlphaSweep] eval alpha={alpha:g} fold={fold} "
                        f"variant={variant} seed={seed}",
                        flush=True,
                    )
                    env = os.environ.copy()
                    env["ISAAC_SKIP_VR_WAIT"] = "1"
                    subprocess.run(command, cwd=PROJECT_DIR, env=env, check=True)
                    missing = [
                        str(path)
                        for path in (output_json, output_csv)
                        if not path.exists()
                    ]
                    if missing:
                        raise RuntimeError(
                            "Evaluation did not write: " + ", ".join(missing)
                        )


def _summarize(
    base_result_dir: Path,
    sweep_dir: Path,
    folds: tuple[int, ...],
    variants: tuple[str, ...],
    alphas: tuple[float, ...],
    seeds: tuple[int, ...],
    reuse_base_alpha: float,
) -> None:
    rows = _load_task_rows(base_result_dir, folds, seeds)
    for alpha in alphas:
        for fold in folds:
            for variant in variants:
                for seed in seeds:
                    if np.isclose(alpha, reuse_base_alpha):
                        path = (
                            base_result_dir
                            / f"fold_{fold:02d}"
                            / f"{variant}_seed_{seed}.json"
                        )
                    else:
                        path = (
                            sweep_dir
                            / _alpha_tag(alpha)
                            / f"fold_{fold:02d}"
                            / f"{variant}_seed_{seed}.json"
                        )
                    rows.extend(_episode_rows(path, fold, seed, variant, alpha))

    sweep_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(sweep_dir / "episode_results.csv", rows)
    session_rows = _aggregate_sessions(rows)
    _write_csv(sweep_dir / "session_summary.csv", session_rows)
    report = _build_report(session_rows, variants, alphas)
    with (sweep_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _write_csv(sweep_dir / "ranking.csv", report["ranking"])
    print(json.dumps(report, indent=2), flush=True)
    print(f"[AlphaSweep] saved summary under {sweep_dir}", flush=True)


def _load_task_rows(
    base_result_dir: Path, folds: tuple[int, ...], seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for seed in seeds:
            path = base_result_dir / f"fold_{fold:02d}" / f"task_seed_{seed}.json"
            rows.extend(_episode_rows(path, fold, seed, "task", None))
    return rows


def _episode_rows(
    path: Path,
    fold: int,
    seed: int,
    variant: str,
    alpha: float | None,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation result: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = []
    for episode in payload["episodes"]:
        steps = max(1, int(episode["steps"]))
        rows.append(
            {
                "fold": fold,
                "variant": variant,
                "alpha": "" if alpha is None else float(alpha),
                "eval_seed": seed,
                "episode": int(episode["episode"]),
                "success": int(bool(episode["success"])),
                "steps": steps,
                "collision_rate": float(episode.get("human_collision_count", 0))
                / steps,
                "near_rate": float(episode.get("near_human_count", 0)) / steps,
                "near_miss_rate": float(episode.get("near_miss_count", 0)) / steps,
                "gate_rate": float(episode.get("safety_gate_active_count", 0)) / steps,
                "min_surface_gap_m": float(
                    episode.get("min_hand_gripper_surface_gap", 10.0)
                ),
                "residual_norm": float(episode.get("mean_safety_residual_norm", 0.0)),
                "applied_position_m": float(
                    episode.get("mean_safety_applied_position_m", 0.0)
                ),
                "max_applied_position_m": float(
                    episode.get("max_safety_applied_position_m", 0.0)
                ),
            }
        )
    return rows


def _aggregate_sessions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted(
        {(int(row["fold"]), str(row["variant"]), row["alpha"]) for row in rows}
    )
    output = []
    for fold, variant, alpha in groups:
        selected = [
            row
            for row in rows
            if row["fold"] == fold
            and row["variant"] == variant
            and row["alpha"] == alpha
        ]
        record: dict[str, Any] = {
            "fold": fold,
            "variant": variant,
            "alpha": alpha,
            "rollouts": len(selected),
        }
        for metric in METRICS:
            record[metric] = float(np.mean([float(row[metric]) for row in selected]))
        output.append(record)
    return output


def _build_report(
    session_rows: list[dict[str, Any]],
    variants: tuple[str, ...],
    alphas: tuple[float, ...],
) -> dict[str, Any]:
    task_by_fold = {
        int(row["fold"]): row for row in session_rows if row["variant"] == "task"
    }
    rng = np.random.default_rng(20260718)
    configurations: list[dict[str, Any]] = []
    ranking: list[dict[str, Any]] = []
    for variant in variants:
        for alpha in alphas:
            selected = [
                row
                for row in session_rows
                if row["variant"] == variant and np.isclose(float(row["alpha"]), alpha)
            ]
            config: dict[str, Any] = {
                "variant": variant,
                "alpha": float(alpha),
                "metrics": {},
            }
            flat: dict[str, Any] = {"variant": variant, "alpha": float(alpha)}
            for metric in METRICS:
                values = np.asarray(
                    [float(row[metric]) for row in selected], dtype=np.float64
                )
                low, high = _bootstrap_ci(values, rng)
                differences = np.asarray(
                    [
                        float(row[metric])
                        - float(task_by_fold[int(row["fold"])][metric])
                        for row in selected
                    ],
                    dtype=np.float64,
                )
                diff_low, diff_high = _bootstrap_ci(differences, rng)
                config["metrics"][metric] = {
                    "mean": float(values.mean()),
                    "ci95": [low, high],
                    "difference_vs_task": float(differences.mean()),
                    "difference_ci95": [diff_low, diff_high],
                }
                flat[metric] = float(values.mean())
                flat[f"{metric}_vs_task"] = float(differences.mean())
            configurations.append(config)
            ranking.append(flat)

    eligible = [row for row in ranking if float(row["success"]) >= 0.95]
    eligible.sort(
        key=lambda row: (
            float(row["collision_rate"]),
            -float(row["min_surface_gap_m"]),
            float(row["near_rate"]),
        )
    )
    for rank, row in enumerate(eligible, start=1):
        row["rank"] = rank
    return {
        "selection_rule": (
            "success >= 0.95, then minimize collision rate, maximize minimum surface gap, "
            "and minimize near rate"
        ),
        "independent_unit": "held-out collection session",
        "session_count": len(task_by_fold),
        "configurations": configurations,
        "ranking": eligible,
    }


def _bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, samples: int = 10000
) -> tuple[float, float]:
    if values.size == 1:
        value = float(values[0])
        return value, value
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _validate_inputs(
    cv_dir: Path,
    policy_dir: Path,
    base_result_dir: Path,
    folds: tuple[int, ...],
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    reuse_base_alpha: float,
) -> None:
    missing = []
    for fold in folds:
        missing.append(cv_dir / f"fold_{fold:02d}" / "eval.hdf5")
        for variant in variants:
            missing.append(
                policy_dir / f"fold_{fold:02d}" / f"ppo_safety_{variant}_best.pt"
            )
        if reuse_base_alpha > 0.0:
            for variant in ("task", *variants):
                for seed in seeds:
                    missing.append(
                        base_result_dir
                        / f"fold_{fold:02d}"
                        / f"{variant}_seed_{seed}.json"
                    )
    missing_paths = [str(path) for path in missing if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing alpha-sweep inputs: " + ", ".join(missing_paths)
        )


def _parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(
        dict.fromkeys(int(value.strip()) for value in text.split(",") if value.strip())
    )
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def _alpha_tag(alpha: float) -> str:
    return "alpha_" + f"{alpha:g}".replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    main()
