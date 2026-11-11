#!/usr/bin/env python3
"""Validate a finalized BC+CBF explicit-feedback HDF5 collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

try:
    from v3_chan.explicit_feedback.validator import validate_collection
    from v3_chan.explicit_feedback.online_schema import ONLINE_SCHEMA_VERSION
    from v3_chan.explicit_feedback.online_validator import validate_online_collection
except ModuleNotFoundError:  # direct execution from inside v3_chan/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from v3_chan.explicit_feedback.validator import validate_collection
    from v3_chan.explicit_feedback.online_schema import ONLINE_SCHEMA_VERSION
    from v3_chan.explicit_feedback.online_validator import validate_online_collection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed validation of finalized legacy or "
            "bc_cbf_online_explicit_feedback_v1 HDF5 data. Public validation "
            "deliberately rejects .partial paths."
        )
    )
    parser.add_argument("path", type=Path, help="finalized .h5 or .hdf5 collection")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one machine-readable JSON report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = _validate_dispatched(args.path)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    elif report.valid:
        if hasattr(report, "trial_count"):
            eligibility = (
                "study-eligible"
                if report.study_eligible
                else "raw/non-study-eligible"
            )
            print(
                f"VALID {eligibility} online collection: {report.path} "
                f"({report.trial_count} trial(s), {report.encounter_count} "
                f"encounter(s), {report.query_count} query(s), "
                f"{report.realtime_marker_count} realtime marker(s))"
            )
        else:
            print(
                f"VALID study-eligible collection: {report.path} "
                f"({report.episode_count} episode(s), "
                f"{report.encounter_count} encounter(s), "
                f"{report.prompt_count} prompt(s), "
                f"{report.response_count} response(s))"
            )
    else:
        print(f"INVALID non-study-eligible collection: {report.path}", file=sys.stderr)
        for issue in report.issues:
            print(
                f"- [{issue.code}] {issue.path}: {issue.message}",
                file=sys.stderr,
            )
    return 0 if report.valid else 1


def _validate_dispatched(path: Path):
    """Dispatch without weakening either versioned validator."""

    schema = ""
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            value = handle.attrs.get("schema_version", "")
            schema = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    except (ImportError, OSError):
        # Let the legacy validator create the normal structured file issue.
        pass
    if schema == ONLINE_SCHEMA_VERSION:
        return validate_online_collection(path, raise_on_error=False)
    return validate_collection(path, raise_on_error=False)


if __name__ == "__main__":
    raise SystemExit(main())
