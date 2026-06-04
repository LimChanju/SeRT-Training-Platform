#!/usr/bin/env python3
"""Switch SteamVR's Linux vrcompositor symlink for ALVR diagnosis.

This is a reversible helper for debugging SteamVR 303 / compositor IPC issues.
It does not delete the ALVR wrapper; it moves the symlink aside as a backup.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path


STEAMVR_BIN = Path(
    "/home/railabchan/.steam/debian-installation/steamapps/common/SteamVR/bin/linux64"
)
VRCOMPOSITOR = STEAMVR_BIN / "vrcompositor"
VRCOMPOSITOR_REAL = STEAMVR_BIN / "vrcompositor.real"
ALVR_WRAPPER = Path(
    "/home/railabchan/.local/share/ALVR-Launcher/installations/v20.14.1/"
    "alvr_streamer_linux/libexec/alvr/vrcompositor-wrapper"
)
BACKUP_PREFIX = "vrcompositor.alvr-wrapper-link.bak-codex-"
REAL_BACKUP_PREFIX = "vrcompositor.real.bak-codex-"


def describe() -> None:
    print(f"SteamVR bin: {STEAMVR_BIN}")
    for path in (VRCOMPOSITOR, VRCOMPOSITOR_REAL):
        if not path.exists() and not path.is_symlink():
            print(f"{path.name}: missing")
            continue
        if path.is_symlink():
            print(f"{path.name}: symlink -> {os.readlink(path)}")
        else:
            print(f"{path.name}: regular file ({path.stat().st_size} bytes)")

    backups = sorted(STEAMVR_BIN.glob(f"{BACKUP_PREFIX}*")) + sorted(
        STEAMVR_BIN.glob(f"{REAL_BACKUP_PREFIX}*")
    )
    if backups:
        print("backups:")
        for backup in backups[-8:]:
            suffix = f" -> {os.readlink(backup)}" if backup.is_symlink() else ""
            print(f"  {backup.name}{suffix}")


def use_original() -> None:
    if not VRCOMPOSITOR_REAL.exists():
        raise SystemExit(f"Missing original compositor: {VRCOMPOSITOR_REAL}")
    if not VRCOMPOSITOR.is_symlink():
        raise SystemExit(
            "Refusing to change vrcompositor because it is not a symlink. "
            "Run --status and inspect it manually."
        )

    target = os.readlink(VRCOMPOSITOR)
    if Path(target).name == "vrcompositor.real":
        print("Already using vrcompositor.real.")
        return

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = STEAMVR_BIN / f"{BACKUP_PREFIX}{stamp}"
    VRCOMPOSITOR.rename(backup)
    VRCOMPOSITOR.symlink_to(VRCOMPOSITOR_REAL)
    print("Temporarily bypassed ALVR compositor wrapper.")
    print(f"Backup: {backup}")
    describe()


def restore_alvr() -> None:
    backups = sorted(STEAMVR_BIN.glob(f"{BACKUP_PREFIX}*"))
    if not backups:
        raise SystemExit("No ALVR wrapper backup found.")
    backup = backups[-1]

    if VRCOMPOSITOR.exists() or VRCOMPOSITOR.is_symlink():
        if VRCOMPOSITOR.is_symlink() and Path(os.readlink(VRCOMPOSITOR)).name == "vrcompositor.real":
            VRCOMPOSITOR.unlink()
        else:
            raise SystemExit(
                "Refusing to overwrite current vrcompositor. "
                "Run --status and inspect it manually."
            )

    backup.rename(VRCOMPOSITOR)
    print("Restored ALVR compositor wrapper.")
    describe()


def install_alvr_wrapper() -> None:
    """Install ALVR wrapper after Steam updates overwrite vrcompositor."""
    if not ALVR_WRAPPER.exists():
        raise SystemExit(f"Missing ALVR wrapper: {ALVR_WRAPPER}")

    if VRCOMPOSITOR.is_symlink() and os.readlink(VRCOMPOSITOR) == str(ALVR_WRAPPER):
        print("Already using ALVR compositor wrapper.")
        describe()
        return

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    if VRCOMPOSITOR.is_file() and not VRCOMPOSITOR.is_symlink():
        if VRCOMPOSITOR_REAL.exists() or VRCOMPOSITOR_REAL.is_symlink():
            real_backup = STEAMVR_BIN / f"{REAL_BACKUP_PREFIX}{stamp}"
            VRCOMPOSITOR_REAL.rename(real_backup)
            print(f"Backed up existing vrcompositor.real: {real_backup}")
        VRCOMPOSITOR.rename(VRCOMPOSITOR_REAL)
        print("Moved current SteamVR compositor to vrcompositor.real.")
    elif VRCOMPOSITOR.is_symlink():
        link_backup = STEAMVR_BIN / f"{BACKUP_PREFIX}{stamp}"
        VRCOMPOSITOR.rename(link_backup)
        print(f"Backed up current vrcompositor symlink: {link_backup}")
    else:
        raise SystemExit(f"Unexpected vrcompositor state: {VRCOMPOSITOR}")

    VRCOMPOSITOR.symlink_to(ALVR_WRAPPER)
    print("Installed ALVR compositor wrapper.")
    describe()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="show current compositor links")
    parser.add_argument(
        "--use-original",
        action="store_true",
        help="temporarily point vrcompositor to SteamVR's original vrcompositor.real",
    )
    parser.add_argument(
        "--restore-alvr",
        action="store_true",
        help="restore the most recent ALVR wrapper symlink backup",
    )
    parser.add_argument(
        "--install-alvr-wrapper",
        action="store_true",
        help="move current vrcompositor to vrcompositor.real and install ALVR wrapper symlink",
    )
    args = parser.parse_args()

    selected = [
        args.status,
        args.use_original,
        args.restore_alvr,
        args.install_alvr_wrapper,
    ].count(True)
    if selected != 1:
        parser.error("choose exactly one of --status, --use-original, --restore-alvr")

    if args.status:
        describe()
    elif args.use_original:
        use_original()
    elif args.restore_alvr:
        restore_alvr()
    elif args.install_alvr_wrapper:
        install_alvr_wrapper()


if __name__ == "__main__":
    main()
