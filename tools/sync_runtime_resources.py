#!/usr/bin/env python3
"""Synchronize repository contracts/configs/specs into installable package resources."""
from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST_ROOT = ROOT / "src" / "nflprops" / "resources"
SOURCES = ("configs", "contracts", "specs")


def _files(root: Path) -> set[Path]:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


def check() -> bool:
    ok = True
    for name in SOURCES:
        src = ROOT / name
        dst = DEST_ROOT / name
        src_files = _files(src)
        dst_files = _files(dst) if dst.exists() else set()
        if src_files != dst_files:
            print(f"{name}: file set differs")
            ok = False
            continue
        for rel in sorted(src_files):
            if src.joinpath(rel).read_bytes() != dst.joinpath(rel).read_bytes():
                print(f"{name}/{rel}: bytes differ")
                ok = False
    return ok


def sync() -> None:
    for name in SOURCES:
        dst = DEST_ROOT / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(ROOT / name, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return 0 if check() else 1
    sync()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
