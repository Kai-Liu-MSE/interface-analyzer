#!/usr/bin/env python3
"""Fit cubic interface parameters from replica-resolved stiffnesses.

Input CSV must have the exact columns ``orientation``, ``replica``, and
``stiffness_mJ_m2``.  Accepted primary orientations are ``100_010``,
``110_001``, and ``110_1m10``.  A complete MD replica is the resampling unit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from interface_analyzer import fit_cubic_replica_blocks


REQUIRED_COLUMNS = ("orientation", "replica", "stiffness_mJ_m2")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _read_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in fields]
        if missing:
            raise ValueError(f"{path} is missing required CSV columns: {', '.join(missing)}")
        records = list(reader)
    if not records:
        raise ValueError(f"{path} has no data rows")
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stiffness_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    if not args.stiffness_csv.is_file():
        parser.error(f"Input CSV does not exist: {args.stiffness_csv}")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be at least 1")

    try:
        result = fit_cubic_replica_blocks(_read_records(args.stiffness_csv), bootstrap=args.bootstrap, seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "replica_combinations.csv", result["replica_combinations"])
    _write_csv(args.output_dir / "replica_block_bootstrap.csv", result["replica_block_bootstrap"])
    _write_csv(args.output_dir / "uncertainty_summary.csv", result["uncertainty_summary"])
    point = _json_ready({
        "schema_version": 1,
        "source_stiffness_csv": args.stiffness_csv.name,
        "primary_orientations": result["primary_orientations"],
        "pooled": result["pooled"],
        "uncertainty_covariance": result["covariance"],
        "replica_uncertainty_policy": "Resample complete MD replicas within each orientation; do not resample Fourier modes.",
        "files": {
            "replica_combinations": "replica_combinations.csv",
            "replica_block_bootstrap": "replica_block_bootstrap.csv",
            "uncertainty_summary": "uncertainty_summary.csv",
        },
    })
    point_path = args.output_dir / "cubic_interface_parameters.json"
    point_path.write_text(json.dumps(point, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(point, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
