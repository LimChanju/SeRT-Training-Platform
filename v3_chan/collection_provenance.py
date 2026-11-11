"""Source-version metadata and production collection validation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any


_SOURCE_SUFFIXES = {".py", ".sh", ".md", ".toml", ".yaml", ".yml"}
_EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    "eval_results",
    "logs",
    "policies",
    "trajectories",
    "videos",
    "gripper_camera_recording",
    "legacy_pre_distal_collider",
}
_INVALID_VERSION_VALUES = {"", "unknown", "unspecified", "none", "null"}
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def source_tree_sha256(project_root: str | os.PathLike[str]) -> str:
    """Hash runnable source files without including datasets or outputs."""

    root = Path(project_root).resolve()
    candidates: list[Path] = []
    launch_script = root / "launch_isaac.sh"
    if launch_script.is_file():
        candidates.append(launch_script)
    source_root = root / "v3_chan"
    if source_root.is_dir():
        for path in source_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _SOURCE_SUFFIXES:
                continue
            if any(part in _EXCLUDED_DIRS for part in path.relative_to(root).parts):
                continue
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"No source files found below {root}")

    digest = hashlib.sha256()
    for path in sorted(set(candidates), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def resolve_code_version(
    project_root: str | os.PathLike[str], explicit_version: str | None = None
) -> tuple[str, str, str]:
    """Return code version, provenance source, and source-tree SHA-256."""

    tree_hash = source_tree_sha256(project_root)
    explicit = str(explicit_version or "").strip()
    if explicit.lower() not in _INVALID_VERSION_VALUES:
        return explicit, "HRI_CODE_VERSION", tree_hash
    return f"source-sha256:{tree_hash}", "source_tree_sha256", tree_hash


def resolve_verified_code_provenance(
    project_root: str | os.PathLike[str],
    declared_commit_sha: str | None = None,
) -> dict[str, Any]:
    """Bind a collection to either a verified clean Git HEAD or source hash.

    A caller-provided SHA is only a consistency assertion; it is never treated
    as proof when the checkout has no Git metadata.  This prevents an old
    handoff SHA (or an arbitrary 40-hex string) from masquerading as the code
    that actually produced a collection.
    """

    root = Path(project_root).expanduser().resolve()
    tree_hash = source_tree_sha256(root)
    declared = str(declared_commit_sha or "").strip().lower()
    if declared and not _COMMIT_RE.fullmatch(declared):
        raise ValueError("declared code commit must be a lowercase 40-hex SHA")

    git_head = ""
    git_clean: bool | None = None
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if Path(top).resolve() != root:
            raise RuntimeError("project root is not the Git toplevel")
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        if not _COMMIT_RE.fullmatch(git_head):
            raise RuntimeError("Git returned an invalid HEAD SHA")
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        git_clean = not bool(status.strip())
    except (FileNotFoundError, OSError, subprocess.CalledProcessError, RuntimeError):
        git_head = ""
        git_clean = None

    if declared and git_head and declared != git_head:
        raise RuntimeError(
            f"declared commit {declared} does not match verified HEAD {git_head}"
        )
    if git_head and git_clean:
        return {
            "code_version": git_head,
            "code_version_source": "verified_git_head_clean",
            "code_commit_sha": git_head,
            "declared_code_commit_sha": declared,
            "code_commit_verification": "verified_git_head_clean",
            "git_worktree_clean": 1,
            "source_tree_sha256": tree_hash,
        }
    source_reason = "git_worktree_dirty" if git_head else "git_metadata_unavailable"
    return {
        "code_version": f"source-sha256:{tree_hash}",
        "code_version_source": "source_tree_sha256",
        "code_commit_sha": git_head,
        "declared_code_commit_sha": declared,
        "code_commit_verification": source_reason,
        "git_worktree_clean": 0 if git_clean is False else -1,
        "source_tree_sha256": tree_hash,
    }


def validate_production_metadata(
    *, production_mode: bool, participant_id: str, code_version: str
) -> None:
    if not production_mode:
        return
    participant = str(participant_id or "").strip().lower()
    version = str(code_version or "").strip().lower()
    if participant in _INVALID_VERSION_VALUES:
        raise RuntimeError(
            "Production collection requires a non-placeholder HRI_PARTICIPANT_ID"
        )
    if version in _INVALID_VERSION_VALUES:
        raise RuntimeError(
            "Production collection requires HRI_CODE_VERSION or a source-tree hash"
        )
