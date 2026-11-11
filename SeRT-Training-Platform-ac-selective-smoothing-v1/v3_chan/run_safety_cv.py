from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAJECTORY_DIR = SCRIPT_DIR / "trajectories"
CV_DIR = TRAJECTORY_DIR / "cv4"
RESULT_DIR = SCRIPT_DIR / "eval_results" / "cv4"
POLICY_DIR = SCRIPT_DIR / "policies" / "cv4"
MANIFEST = TRAJECTORY_DIR / "trajectory_quality_manifest.json"
TASK_CHECKPOINT = SCRIPT_DIR / "policies" / "ppo_pick_place_v7_residual_rewardv4_strict_best.pt"
SESSION_FILES = (
    TRAJECTORY_DIR / "hri_surface_gap_session_01.hdf5",
    TRAJECTORY_DIR / "hri_surface_gap_session_02.hdf5",
    TRAJECTORY_DIR / "hri_surface_gap_session_03.hdf5",
    TRAJECTORY_DIR / "hri_surface_gap_session_04.hdf5",
)
VARIANTS = ("physical", "proxy")
V4_SESSION_SCHEMA = "hri_obs_v4_builtin_panda_collision_geometry"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run session-level 4-fold HRI safety-policy CV.")
    parser.add_argument("--stage", choices=("prepare", "train", "eval", "summarize", "all"), default="all")
    parser.add_argument("--folds", default="1,2,3,4")
    parser.add_argument("--variants", default="physical,proxy")
    parser.add_argument(
        "--session-files",
        nargs="+",
        default=[str(path) for path in SESSION_FILES],
        help="Four v4 HRI session HDF5 files, one held-out session per fold.",
    )
    parser.add_argument("--total-steps", type=int, default=30000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument(
        "--eval-seeds",
        default="11,1011,2011,3011,4011,5011,6011,7011,8011,9011",
        help="Comma-separated rollout seeds; ten seeds are used by default for the final comparison.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--residual-alpha", type=float, default=0.1)
    parser.add_argument("--safety-gate-start-dist", type=float, default=0.13)
    parser.add_argument("--safety-gate-full-dist", type=float, default=0.05)
    parser.add_argument(
        "--gate-active-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train PPO only on transitions with a positive distance gate.",
    )
    parser.add_argument(
        "--distance-progress-weight",
        type=float,
        default=2.0,
        help="Reward weight for increased hand-gripper clearance while the gate is active.",
    )
    parser.add_argument("--distance-progress-clip-m", type=float, default=0.03)
    parser.add_argument(
        "--xyz-only-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict the learned safety residual to XYZ; yaw and gripper residuals remain zero.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--experiment-tag",
        default="surface_gap_new_sessions_v1",
        help="Separate output directory for this CV configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    global CV_DIR, RESULT_DIR, POLICY_DIR, SESSION_FILES
    SESSION_FILES = tuple(Path(path).resolve() for path in args.session_files)
    if len(SESSION_FILES) != 4:
        raise ValueError(
            f"Session-level 4-fold CV requires exactly four files; got {len(SESSION_FILES)}"
        )
    experiment_tag = args.experiment_tag.strip()
    if not experiment_tag or any(char in experiment_tag for char in "/\\"):
        raise ValueError("--experiment-tag must be a non-empty directory name")
    RESULT_DIR = SCRIPT_DIR / "eval_results" / "cv4" / experiment_tag
    POLICY_DIR = SCRIPT_DIR / "policies" / "cv4" / experiment_tag
    CV_DIR = TRAJECTORY_DIR / "cv4" / experiment_tag
    folds = _parse_int_set(args.folds, 1, len(SESSION_FILES))
    variants = _parse_variants(args.variants)
    eval_seeds = tuple(int(value.strip()) for value in args.eval_seeds.split(",") if value.strip())
    if not eval_seeds:
        raise ValueError("--eval-seeds must contain at least one integer")
    _validate_inputs()

    if args.stage in ("prepare", "all"):
        _prepare_folds(folds, force=args.force)
    if args.stage in ("train", "all"):
        _require_prepared(folds)
        _train_folds(folds, variants, args)
    if args.stage in ("eval", "all"):
        _require_prepared(folds)
        _evaluate_folds(folds, variants, eval_seeds, args)
    if args.stage in ("summarize", "all"):
        _summarize(folds, variants, eval_seeds)


def _prepare_folds(folds: tuple[int, ...], *, force: bool) -> None:
    for fold in folds:
        fold_dir = CV_DIR / f"fold_{fold:02d}"
        train_output = fold_dir / "train.hdf5"
        eval_output = fold_dir / "eval.hdf5"
        train_inputs = [path for index, path in enumerate(SESSION_FILES, start=1) if index != fold]
        eval_input = SESSION_FILES[fold - 1]
        if force or not train_output.exists():
            _prepare_dataset(train_output, train_inputs, include_verified_failures=False)
        else:
            print(f"[SafetyCV] skip existing {train_output}", flush=True)
        if force or not eval_output.exists():
            _prepare_dataset(eval_output, [eval_input], include_verified_failures=True)
        else:
            print(f"[SafetyCV] skip existing {eval_output}", flush=True)
        print(
            f"[SafetyCV] fold={fold} train_eps={_episode_count(train_output)} "
            f"eval_eps={_episode_count(eval_output)} held_out={eval_input.name}",
            flush=True,
        )


def _prepare_dataset(output: Path, inputs: list[Path], *, include_verified_failures: bool) -> None:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_hri_dataset.py"),
        "--output",
        str(output),
        "--manifest",
        str(MANIFEST),
    ]
    if include_verified_failures:
        command.append("--include-verified-failures")
    command.extend(str(path) for path in inputs)
    _run(command)


def _train_folds(folds: tuple[int, ...], variants: tuple[str, ...], args: argparse.Namespace) -> None:
    for fold in folds:
        train_data = CV_DIR / f"fold_{fold:02d}" / "train.hdf5"
        for variant in variants:
            output = _policy_path(fold, variant, best=False)
            best_output = _policy_path(fold, variant, best=True)
            if not args.force and output.exists() and best_output.exists():
                print(f"[SafetyCV] skip trained fold={fold} variant={variant}", flush=True)
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
                str(args.distance_progress_weight),
                "--distance-progress-clip-m",
                str(args.distance_progress_clip_m),
                "--seed",
                str(23 + fold),
            )
            if args.gate_active_only:
                command.append("--gate-active-only")
            if args.xyz_only_residual:
                command.append("--xyz-only-residual")
            command.append("--no-pseudo-errp" if variant == "physical" else "--pseudo-errp")
            print(f"[SafetyCV] train fold={fold} variant={variant}", flush=True)
            _run(command, isaac=True)


def _evaluate_folds(
    folds: tuple[int, ...],
    variants: tuple[str, ...],
    eval_seeds: tuple[int, ...],
    args: argparse.Namespace,
) -> None:
    for fold in folds:
        eval_data = CV_DIR / f"fold_{fold:02d}" / "eval.hdf5"
        episode_count = _episode_count(eval_data)
        models = ("task",) + variants
        for model in models:
            safety_checkpoint = None if model == "task" else _policy_path(fold, model, best=True)
            if safety_checkpoint is not None and not safety_checkpoint.exists():
                raise FileNotFoundError(f"Missing trained checkpoint: {safety_checkpoint}")
            for seed in eval_seeds:
                output_json = _eval_path(fold, model, seed, suffix="json")
                output_csv = _eval_path(fold, model, seed, suffix="csv")
                if not args.force and output_json.exists() and output_csv.exists():
                    print(f"[SafetyCV] skip evaluated fold={fold} model={model} seed={seed}", flush=True)
                    continue
                output_json.parent.mkdir(parents=True, exist_ok=True)
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
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(output_csv),
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
                print(f"[SafetyCV] eval fold={fold} model={model} seed={seed}", flush=True)
                _run(command, isaac=True)
                missing_outputs = [
                    str(path) for path in (output_json, output_csv) if not path.exists()
                ]
                if missing_outputs:
                    raise RuntimeError(
                        "Evaluation exited without writing required outputs: "
                        + ", ".join(missing_outputs)
                    )


def _summarize(folds: tuple[int, ...], variants: tuple[str, ...], eval_seeds: tuple[int, ...]) -> None:
    models = ("task",) + variants
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for model in models:
            for seed in eval_seeds:
                path = _eval_path(fold, model, seed, suffix="json")
                if not path.exists():
                    raise FileNotFoundError(f"Missing evaluation result: {path}")
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                for episode in payload["episodes"]:
                    steps = max(1, int(episode["steps"]))
                    rows.append(
                        {
                            "fold": fold,
                            "model": model,
                            "eval_seed": seed,
                            "episode": int(episode["episode"]),
                            "success": int(bool(episode["success"])),
                            "steps": steps,
                            "total_reward": float(episode["total_reward"]),
                            "errp_count": int(episode.get("errp_count", 0)),
                            "gate_rate": float(episode.get("safety_gate_active_count", 0)) / steps,
                            "near_rate": float(episode.get("near_human_count", 0)) / steps,
                            "near_miss_rate": float(episode.get("near_miss_count", 0)) / steps,
                            "collision_rate": float(episode.get("human_collision_count", 0)) / steps,
                            "min_hand_gripper_dist": float(episode.get("min_hand_gripper_dist", 10.0)),
                            "residual_norm": float(episode.get("mean_safety_residual_norm", 0.0)),
                        }
                    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(RESULT_DIR / "episode_results.csv", rows)
    aggregate_rows = _aggregate_session_rows(rows, models)
    _write_csv(RESULT_DIR / "session_summary.csv", aggregate_rows)
    report = _build_report(aggregate_rows, models)
    with (RESULT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    print(f"[SafetyCV] saved summaries under {RESULT_DIR}", flush=True)


def _aggregate_session_rows(rows: list[dict[str, Any]], models: tuple[str, ...]) -> list[dict[str, Any]]:
    metrics = ("success", "steps", "total_reward", "errp_count", "gate_rate", "near_rate", "near_miss_rate", "collision_rate", "min_hand_gripper_dist", "residual_norm")
    output = []
    for fold in sorted({int(row["fold"]) for row in rows}):
        for model in models:
            selected = [row for row in rows if row["fold"] == fold and row["model"] == model]
            record: dict[str, Any] = {"fold": fold, "model": model, "rollouts": len(selected)}
            for metric in metrics:
                record[metric] = float(np.mean([float(row[metric]) for row in selected]))
            output.append(record)
    return output


def _build_report(session_rows: list[dict[str, Any]], models: tuple[str, ...]) -> dict[str, Any]:
    metrics = ("success", "steps", "total_reward", "errp_count", "gate_rate", "near_rate", "near_miss_rate", "collision_rate", "min_hand_gripper_dist", "residual_norm")
    rng = np.random.default_rng(20260715)
    report: dict[str, Any] = {"independent_unit": "held-out collection session", "session_count": len({row["fold"] for row in session_rows}), "models": {}}
    by_model = {model: [row for row in session_rows if row["model"] == model] for model in models}
    for model in models:
        model_report: dict[str, Any] = {}
        for metric in metrics:
            values = np.asarray([row[metric] for row in by_model[model]], dtype=np.float64)
            low, high = _bootstrap_ci(values, rng)
            model_report[metric] = {"mean": float(values.mean()), "ci95": [low, high]}
        report["models"][model] = model_report
    task_by_fold = {row["fold"]: row for row in by_model["task"]}
    report["paired_difference_vs_task"] = {}
    for model in models:
        if model == "task":
            continue
        differences: dict[str, Any] = {}
        for metric in metrics:
            values = np.asarray(
                [row[metric] - task_by_fold[row["fold"]][metric] for row in by_model[model]],
                dtype=np.float64,
            )
            low, high = _bootstrap_ci(values, rng)
            differences[metric] = {"mean": float(values.mean()), "ci95": [low, high]}
        report["paired_difference_vs_task"][model] = differences
    return report


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int = 10000) -> tuple[float, float]:
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


def _policy_path(fold: int, variant: str, *, best: bool) -> Path:
    suffix = "_best" if best else ""
    return POLICY_DIR / f"fold_{fold:02d}" / f"ppo_safety_{variant}{suffix}.pt"


def _eval_path(fold: int, model: str, seed: int, *, suffix: str) -> Path:
    return RESULT_DIR / f"fold_{fold:02d}" / f"{model}_seed_{seed}.{suffix}"


def _isaac_command(script: Path, *arguments: str) -> list[str]:
    return [str(PROJECT_DIR / "launch_isaac.sh"), str(script), *arguments]


def _run(command: list[str], *, isaac: bool = False) -> None:
    env = os.environ.copy()
    if isaac:
        env["ISAAC_SKIP_VR_WAIT"] = "1"
    subprocess.run(command, cwd=PROJECT_DIR, env=env, check=True)


def _episode_count(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return len(f["episodes"])


def _require_prepared(folds: tuple[int, ...]) -> None:
    missing = []
    for fold in folds:
        for name in ("train.hdf5", "eval.hdf5"):
            path = CV_DIR / f"fold_{fold:02d}" / name
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Prepare CV folds first: {missing}")


def _validate_inputs() -> None:
    missing = [str(path) for path in (*SESSION_FILES, TASK_CHECKPOINT) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required CV inputs: {missing}")
    for path in SESSION_FILES:
        _validate_v4_session(path)


def _validate_v4_session(path: Path) -> None:
    with h5py.File(path, "r") as source:
        schema = str(source.attrs.get("schema_version", ""))
        if schema != V4_SESSION_SCHEMA:
            raise ValueError(
                f"{path} uses schema '{schema}', expected '{V4_SESSION_SCHEMA}'. "
                "Do not mix pre-v4 trajectories into this CV run."
            )
        if "episodes" not in source or len(source["episodes"]) == 0:
            raise ValueError(f"{path} contains no recorded episodes")
        for episode_name, episode in source["episodes"].items():
            if "human" not in episode:
                raise ValueError(f"{path}:{episode_name} has no human trajectory group")
            human = episode["human"]
            required = ("head_pos", "left_hand_pos", "right_hand_pos")
            missing_fields = [name for name in required if name not in human]
            if missing_fields:
                raise ValueError(
                    f"{path}:{episode_name} is missing human fields {missing_fields}"
                )
            lengths = {int(human[name].shape[0]) for name in required}
            if len(lengths) != 1 or next(iter(lengths)) <= 0:
                raise ValueError(
                    f"{path}:{episode_name} has inconsistent or empty human trajectories"
                )


def _parse_int_set(text: str, low: int, high: int) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(value.strip()) for value in text.split(",") if value.strip()))
    if not values or any(value < low or value > high for value in values):
        raise ValueError(f"Values must be within [{low}, {high}]: {text}")
    return values


def _parse_variants(text: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(value.strip() for value in text.split(",") if value.strip()))
    unknown = sorted(set(values) - set(VARIANTS))
    if not values or unknown:
        raise ValueError(f"Unknown variants: {unknown}; expected {VARIANTS}")
    return values


if __name__ == "__main__":
    main()
