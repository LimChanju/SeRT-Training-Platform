from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge HRI session files while preserving source provenance."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument(
        "--include-verified-failures",
        action="store_true",
        help="Include episodes marked verified_success=false in the quality manifest.",
    )
    parser.add_argument("inputs", nargs="+")
    return parser.parse_args()


def _load_exclusions(path: str) -> set[tuple[str, str]]:
    if not path:
        return set()
    if not os.path.exists(path):
        print(f"Quality manifest not found; no episodes will be excluded: {path}")
        return set()
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    exclusions = set()
    for row in manifest.get("episode_labels", []):
        if row.get("verified_success") is False:
            exclusions.add((str(row["file"]), str(row["episode"])))
    return exclusions


def _copy_root_attrs(source: h5py.File, output: h5py.File) -> None:
    for key, value in source.attrs.items():
        output.attrs[key] = value
    output.attrs["prepared_dataset"] = True
    output.attrs["preparation_schema"] = "hri_session_merge_v1"


def main() -> None:
    args = _parse_args()
    inputs = [Path(path).resolve() for path in args.inputs]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    exclusions = set() if args.include_verified_failures else _load_exclusions(args.manifest)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = []
    with h5py.File(output_path, "w") as output:
        output_episodes = output.create_group("episodes")
        root_attrs_written = False
        for input_path in inputs:
            with h5py.File(input_path, "r") as source:
                if not root_attrs_written:
                    _copy_root_attrs(source, output)
                    root_attrs_written = True
                if "episodes" not in source:
                    continue
                for source_episode_name in sorted(source["episodes"].keys()):
                    key = (input_path.name, source_episode_name)
                    if key in exclusions:
                        skipped.append(key)
                        continue
                    output_name = f"episode_{copied:06d}"
                    source.copy(source["episodes"][source_episode_name], output_episodes, name=output_name)
                    target = output_episodes[output_name]
                    target.attrs["source_file"] = input_path.name
                    target.attrs["source_episode"] = source_episode_name
                    copied += 1

        output.attrs["episode_count"] = copied
        output.attrs["source_files"] = json.dumps([path.name for path in inputs])
        output.attrs["excluded_episode_count"] = len(skipped)

    print(f"Prepared {output_path}")
    print(f"Copied episodes: {copied}")
    print(f"Skipped episodes: {len(skipped)}")
    for file_name, episode_name in skipped:
        print(f"  skipped {file_name}:{episode_name}")


if __name__ == "__main__":
    main()
