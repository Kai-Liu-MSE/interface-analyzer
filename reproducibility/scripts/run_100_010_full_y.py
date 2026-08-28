#!/usr/bin/env python3
"""Run the documented v2 full-Y smoke workflow on the bundled 100[010] frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from interface_analyzer import cfm_spectrum, extract_trajectory, fit_cfm_tensor, write_interface_pickle
from verify_100_010_dataset import DATA_DIR, load_manifest, verify_dataset


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RESULTS = BUNDLE_ROOT / "expected_results" / "100_010_20frames_full_y.json"
EXTRACTION_PARAMETERS = {
    "mode": "full_y",
    "descriptor": "orientation_phi",
    "lattice_constant": 4.134,
    "miller_x": (1, 0, 0),
    "miller_y": (0, 1, 0),
    "miller_z": (0, 0, 1),
    "xz_grid": 2.5,
    "radius": 6.0,
    "n": 5,
    "save_field": False,
}


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


def build_summary(results: dict[int | str, dict[str, object]], temperature: float) -> dict[str, Any]:
    steps = sorted((int(step) for step in results))
    first, last = results[steps[0]], results[steps[-1]]
    upper_means = np.asarray([np.mean(np.asarray(results[step]["h_upper"], dtype=float)) for step in steps])
    lower_means = np.asarray([np.mean(np.asarray(results[step]["h_lower"], dtype=float)) for step in steps])
    spectrum = cfm_spectrum(results, temperature=temperature)
    fit = fit_cfm_tensor(spectrum, model="ky0", k2_min=0.005, k2_max=0.03)
    return _json_ready({
        "bundle_schema_version": 1,
        "dataset": DATA_DIR.name,
        "n_frames": len(steps),
        "steps": steps,
        "extraction_parameters": EXTRACTION_PARAMETERS,
        "interface": {
            "grid_shape_zx": list(first["grid_shape"]),
            "height_shape_x": list(np.asarray(first["h_upper"]).shape),
            "first_h_upper_mean_A": float(upper_means[0]),
            "first_h_lower_mean_A": float(lower_means[0]),
            "last_h_upper_mean_A": float(upper_means[-1]),
            "last_h_lower_mean_A": float(lower_means[-1]),
            "trajectory_h_upper_mean_A": float(np.mean(upper_means)),
            "trajectory_h_lower_mean_A": float(np.mean(lower_means)),
        },
        "native_grid_cfm_smoke": {
            "temperature_K": float(temperature),
            "n_independent_modes": int(len(spectrum["k2_Ainv2"])),
            "projection_identity_max_abs_difference_A": float(spectrum["projection_identity_max_abs_difference_A"]),
            "ky0_fit": fit,
        },
    })


def _get(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for component in dotted_key.split("."):
        value = value[component]
    return value


def check_reference(summary: dict[str, Any], path: Path = EXPECTED_RESULTS) -> None:
    reference = json.loads(path.read_text(encoding="utf-8"))
    tolerance = reference["comparison_tolerance"]
    for dotted_key, expected in reference["exact"].items():
        observed = _get(summary, dotted_key)
        if observed != expected:
            raise AssertionError(f"{dotted_key}: observed {observed!r}, expected {expected!r}")
    for dotted_key, expected in reference["floating_point"].items():
        observed = float(_get(summary, dotted_key))
        if not np.isclose(observed, float(expected), rtol=float(tolerance["rtol"]), atol=float(tolerance["atol"])):
            raise AssertionError(f"{dotted_key}: observed {observed:.12g}, expected {float(expected):.12g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for generated pkl.gz and summary JSON.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=927.0)
    parser.add_argument("--check-reference", action="store_true")
    args = parser.parse_args()
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")

    verification = verify_dataset()
    paths = [DATA_DIR / row["filename"] for row in load_manifest()]
    results = extract_trajectory(paths, workers=args.workers, **EXTRACTION_PARAMETERS)
    summary = build_summary(results, args.temperature)
    summary["input_verification"] = verification

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = write_interface_pickle(results, args.output_dir / "100_010_v2_full_y.pkl.gz")
    summary["output_pickle"] = pickle_path.name
    summary_path = args.output_dir / "100_010_v2_full_y_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.check_reference:
        check_reference(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
