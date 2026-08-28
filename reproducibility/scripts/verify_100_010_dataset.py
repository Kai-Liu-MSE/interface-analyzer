#!/usr/bin/env python3
"""Verify the immutable 100[010] smoke-test dataset before processing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = BUNDLE_ROOT / "data" / "100_010_20frames"
MANIFEST = DATA_DIR / "manifest.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _header(path: Path) -> tuple[int, int]:
    with path.open("rt", encoding="ascii") as handle:
        lines = [handle.readline().strip() for _ in range(10)]
    if lines[0] != "ITEM: TIMESTEP" or lines[2] != "ITEM: NUMBER OF ATOMS":
        raise ValueError(f"{path.name}: not a recognized LAMMPS custom dump")
    if lines[4] != "ITEM: BOX BOUNDS pp pp pp":
        raise ValueError(f"{path.name}: expected periodic orthorhombic box header")
    if not lines[8].startswith("ITEM: ATOMS id type "):
        raise ValueError(f"{path.name}: expected id/type/coordinate atom columns")
    return int(lines[1]), int(lines[3])


def load_manifest(path: Path = MANIFEST) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"filename", "timestep", "bytes", "sha256"}
    if not rows or set(rows[0]) != required:
        raise ValueError(f"{path}: expected CSV columns {sorted(required)}")
    return rows


def verify_dataset(*, check_hashes: bool = True) -> dict[str, Any]:
    """Verify all declared files and return only portable summary metadata."""
    rows = load_manifest()
    expected_steps = list(range(1_000_000, 1_009_501, 500))
    if [int(row["timestep"]) for row in rows] != expected_steps:
        raise ValueError("Manifest does not declare the expected 20-frame, 500-step sequence")

    atom_counts: set[int] = set()
    for row in rows:
        path = DATA_DIR / row["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"{path.name}: byte count differs from manifest")
        timestep, atom_count = _header(path)
        if timestep != int(row["timestep"]):
            raise ValueError(f"{path.name}: header timestep differs from manifest")
        atom_counts.add(atom_count)
        if check_hashes and _sha256(path) != row["sha256"]:
            raise ValueError(f"{path.name}: SHA-256 differs from manifest")

    return {
        "dataset": DATA_DIR.name,
        "n_frames": len(rows),
        "step_start": expected_steps[0],
        "step_end": expected_steps[-1],
        "step_interval": 500,
        "atom_counts": sorted(atom_counts),
        "sha256_checked": check_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hashes", action="store_true", help="Check metadata and headers but not SHA-256 values.")
    args = parser.parse_args()
    print(json.dumps(verify_dataset(check_hashes=not args.skip_hashes), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
