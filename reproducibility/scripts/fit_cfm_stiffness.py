#!/usr/bin/env python3
"""Fit a native-grid CFM stiffness from an interface-height pickle.

The input must have been produced by ``interface-analyzer-extract`` or the
bundled full-Y post-processing script.  This program writes the selected raw
Fourier modes as CSV and the through-origin stiffness fit as JSON; it does not
modify the input pickle.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from interface_analyzer import cfm_spectrum, fit_cfm_tensor, load_interface_pickle


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


def _write_modes(path: Path, spectrum: dict[str, Any], *, k2_min: float, k2_max: float, model: str) -> int:
    k2 = np.asarray(spectrum["k2_Ainv2"], dtype=float)
    selected = np.isfinite(np.asarray(spectrum["response_mJ_m2_Ainv2"], dtype=float)) & (k2 > k2_min) & (k2 < k2_max)
    if model == "ky0":
        selected &= np.asarray(spectrum["ny"], dtype=int) == 0
    elif model == "kx0":
        selected &= np.asarray(spectrum["nx"], dtype=int) == 0
    columns = (
        "nx", "ny", "kx_Ainv", "ky_Ainv", "k2_Ainv2", "theta_deg",
        "power_upper_A2", "power_lower_A2", "power_combined_A2", "response_mJ_m2_Ainv2",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("selected_by_fit", *columns))
        writer.writeheader()
        for index in range(len(k2)):
            row = {column: _json_ready(np.asarray(spectrum[column])[index]) for column in columns}
            row["selected_by_fit"] = bool(selected[index])
            writer.writerow(row)
    return int(np.count_nonzero(selected))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interface_pickle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True, help="Measured trajectory temperature in K.")
    parser.add_argument("--model", choices=("ky0", "kx0", "diagonal", "full", "isotropic"), default="ky0")
    parser.add_argument("--k2-min", type=float, default=0.005, metavar="A_INV2")
    parser.add_argument("--k2-max", type=float, default=0.03, metavar="A_INV2")
    args = parser.parse_args()
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")
    if args.k2_min < 0.0 or args.k2_max <= args.k2_min:
        parser.error("Require 0 <= --k2-min < --k2-max")
    if not args.interface_pickle.is_file():
        parser.error(f"Input pickle does not exist: {args.interface_pickle}")

    results = load_interface_pickle(args.interface_pickle)
    spectrum = cfm_spectrum(results, temperature=args.temperature)
    fit = fit_cfm_tensor(spectrum, model=args.model, k2_min=args.k2_min, k2_max=args.k2_max)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes_path = args.output_dir / "native_grid_cfm_modes.csv"
    n_selected = _write_modes(modes_path, spectrum, k2_min=args.k2_min, k2_max=args.k2_max, model=args.model)
    summary = _json_ready({
        "schema_version": 1,
        "source_interface_pickle": args.interface_pickle.name,
        "temperature_K": args.temperature,
        "fit_window_k2_Ainv2": {"exclusive_min": args.k2_min, "exclusive_max": args.k2_max},
        "fit": fit,
        "n_independent_modes": len(spectrum["k2_Ainv2"]),
        "n_selected_modes": n_selected,
        "projection_identity_max_abs_difference_A": spectrum["projection_identity_max_abs_difference_A"],
        "modes_csv": modes_path.name,
    })
    summary_path = args.output_dir / "cfm_stiffness.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if fit["status"] != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
