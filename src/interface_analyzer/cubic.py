"""Cubic interfacial-stiffness fitting with replica-aware uncertainty.

The parameterization follows the established solid--liquid-interface CFM
convention.  Directional stiffnesses are written as a linear function of
``p = (gamma0, gamma0 * epsilon1, gamma0 * epsilon2)``.  Statistical
resampling is always at the *complete MD replica* level; Fourier modes within
one trajectory must not be treated as independent replicas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np


# The three directions are sufficient to determine the conventional cubic
# parameterization.  Keep these machine-readable names in CSV input files.
PRIMARY_ORIENTATIONS = ("100_010", "110_001", "110_1m10")
CUBIC_COEFFICIENTS = {
    "100_010": np.array((1.0, -18.0 / 5.0, -80.0 / 7.0)),
    "110_001": np.array((1.0, -21.0 / 10.0, 365.0 / 14.0)),
    "110_1m10": np.array((1.0, 39.0 / 10.0, 155.0 / 14.0)),
}


def _as_finite_float(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _fit_result(design: np.ndarray, response: np.ndarray) -> dict[str, float]:
    if design.ndim != 2 or design.shape[1] != 3:
        raise ValueError("Cubic design matrix must have exactly three columns")
    if len(response) < 3 or np.linalg.matrix_rank(design) < 3:
        raise ValueError("At least three linearly independent directional stiffnesses are required")
    parameters, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    gamma0, gamma0_epsilon1, gamma0_epsilon2 = map(float, parameters)
    if np.isclose(gamma0, 0.0):
        raise ValueError("gamma0 is zero, so epsilon1 and epsilon2 are undefined")
    prediction = design @ parameters
    residual = response - prediction
    total = float(np.sum((response - np.mean(response)) ** 2))
    result: dict[str, float] = {
        "gamma0_mJ_m2": gamma0,
        "epsilon1": gamma0_epsilon1 / gamma0,
        "epsilon2": gamma0_epsilon2 / gamma0,
        "minus_epsilon2": -gamma0_epsilon2 / gamma0,
        "gamma0_epsilon1_mJ_m2": gamma0_epsilon1,
        "gamma0_epsilon2_mJ_m2": gamma0_epsilon2,
        "n_directional_stiffnesses": int(len(response)),
        "rmse_mJ_m2": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / total) if total else float("nan"),
        "condition_number": float(np.linalg.cond(design)),
    }
    result.update(_directional_stiffnesses(parameters))
    return result


def _directional_stiffnesses(parameters: np.ndarray) -> dict[str, float]:
    beta_100 = float(CUBIC_COEFFICIENTS["100_010"] @ parameters)
    beta_001 = float(CUBIC_COEFFICIENTS["110_001"] @ parameters)
    beta_1m10 = float(CUBIC_COEFFICIENTS["110_1m10"] @ parameters)
    return {
        "beta_100_010_mJ_m2": beta_100,
        "beta_110_001_mJ_m2": beta_001,
        "beta_110_1m10_mJ_m2": beta_1m10,
        "beta_110_1m12_mJ_m2": (2.0 * beta_001 + beta_1m10) / 3.0,
        "beta_110_1m1m1_mJ_m2": (beta_001 + 2.0 * beta_1m10) / 3.0,
        "gamma_xy_110_1m12_mJ_m2": np.sqrt(2.0) * (beta_1m10 - beta_001) / 3.0,
    }


def fit_cubic_stiffness(
    stiffness_by_orientation: Mapping[str, object], *, orientations: Sequence[str] = PRIMARY_ORIENTATIONS
) -> dict[str, float]:
    """Fit ``gamma0``, ``epsilon1`` and ``epsilon2`` from directional stiffnesses.

    Parameters use mJ m^-2.  Supplying more than three registered orientations
    performs an unweighted least-squares fit; the standard three-orientation
    input is an exact linear inversion up to floating-point rounding.
    """

    selected = tuple(orientations)
    if len(selected) < 3:
        raise ValueError("At least three orientations are required")
    unknown = [orientation for orientation in selected if orientation not in CUBIC_COEFFICIENTS]
    if unknown:
        raise ValueError(f"Unknown cubic orientation(s): {', '.join(unknown)}")
    missing = [orientation for orientation in selected if orientation not in stiffness_by_orientation]
    if missing:
        raise ValueError(f"Missing stiffness for orientation(s): {', '.join(missing)}")
    design = np.vstack([CUBIC_COEFFICIENTS[orientation] for orientation in selected])
    response = np.asarray(
        [_as_finite_float(stiffness_by_orientation[orientation], name=f"stiffness[{orientation}]") for orientation in selected],
        dtype=float,
    )
    return _fit_result(design, response)


def _distribution_statistics(rows: Sequence[Mapping[str, object]], quantities: Sequence[str]) -> tuple[list[dict[str, float | int | str]], dict[str, Any]]:
    summary: list[dict[str, float | int | str]] = []
    vectors: list[np.ndarray] = []
    for quantity in quantities:
        values = np.asarray([float(row[quantity]) for row in rows], dtype=float)
        vectors.append(values)
        sample_std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        summary.append({
            "quantity": quantity,
            "mean": float(np.mean(values)),
            "sample_std": sample_std,
            "ci_2p5": float(np.quantile(values, 0.025)),
            "ci_97p5": float(np.quantile(values, 0.975)),
            "n": int(len(values)),
        })
    covariance = np.cov(np.vstack(vectors), ddof=1).tolist() if len(rows) > 1 else []
    return summary, {"quantities": list(quantities), "matrix": covariance, "n": int(len(rows))}


def fit_cubic_replica_blocks(
    records: Sequence[Mapping[str, object]], *, bootstrap: int = 10000, seed: int = 20260828,
    orientations: Sequence[str] = PRIMARY_ORIENTATIONS,
) -> dict[str, Any]:
    """Propagate directional MD-replica uncertainty into cubic parameters.

    Every record must contain ``orientation``, ``replica`` and
    ``stiffness_mJ_m2``.  The point estimate fits the mean stiffness of each
    direction.  The returned `replica_combinations` selects one whole replica
    per direction, while `replica_block_bootstrap` resamples complete replicas
    with replacement within each direction.  No individual Fourier mode is
    ever resampled here.
    """

    if bootstrap < 1:
        raise ValueError("bootstrap must be at least 1")
    selected = tuple(orientations)
    if set(selected) != set(PRIMARY_ORIENTATIONS) or len(selected) != len(PRIMARY_ORIENTATIONS):
        raise ValueError(f"Replica-block fitting requires exactly {PRIMARY_ORIENTATIONS}")
    groups: dict[str, list[tuple[str, float]]] = {orientation: [] for orientation in selected}
    seen: set[tuple[str, str]] = set()
    for row_number, record in enumerate(records, start=1):
        try:
            orientation = str(record["orientation"])
            replica = str(record["replica"])
            stiffness = _as_finite_float(record["stiffness_mJ_m2"], name=f"row {row_number} stiffness_mJ_m2")
        except KeyError as exc:
            raise ValueError(f"row {row_number} is missing required column {exc.args[0]!r}") from exc
        if orientation not in groups:
            raise ValueError(f"row {row_number} has unsupported orientation {orientation!r}")
        key = (orientation, replica)
        if key in seen:
            raise ValueError(f"duplicate replica {replica!r} for orientation {orientation!r}")
        seen.add(key)
        groups[orientation].append((replica, stiffness))
    missing = [orientation for orientation, group in groups.items() if not group]
    if missing:
        raise ValueError(f"No replicas supplied for orientation(s): {', '.join(missing)}")

    for orientation in selected:
        groups[orientation].sort(key=lambda item: item[0])
    orientation_means = {orientation: float(np.mean([value for _, value in groups[orientation]])) for orientation in selected}
    pooled = fit_cubic_stiffness(orientation_means, orientations=selected)
    pooled.update({
        "estimator": "unweighted cubic fit of each orientation's replica mean",
        "orientation_mean_stiffness_mJ_m2": orientation_means,
        "n_replicas_by_orientation": {orientation: len(groups[orientation]) for orientation in selected},
    })

    combination_rows: list[dict[str, Any]] = []
    for selected_replicas in product(*(groups[orientation] for orientation in selected)):
        values = {orientation: value for orientation, (_, value) in zip(selected, selected_replicas)}
        row = {
            f"replica_{orientation}": replica for orientation, (replica, _) in zip(selected, selected_replicas)
        }
        row.update(fit_cubic_stiffness(values, orientations=selected))
        combination_rows.append(row)

    rng = np.random.default_rng(seed)
    bootstrap_rows: list[dict[str, Any]] = []
    for iteration in range(1, bootstrap + 1):
        resampled_means: dict[str, float] = {}
        row: dict[str, Any] = {"iteration": iteration}
        for orientation in selected:
            values = np.asarray([value for _, value in groups[orientation]], dtype=float)
            indices = rng.integers(0, len(values), size=len(values))
            resampled_means[orientation] = float(np.mean(values[indices]))
            row[f"resampled_replicas_{orientation}"] = ";".join(groups[orientation][index][0] for index in indices)
        row.update(fit_cubic_stiffness(resampled_means, orientations=selected))
        bootstrap_rows.append(row)

    quantities = (
        "gamma0_mJ_m2", "epsilon1", "epsilon2", "minus_epsilon2",
        "beta_100_010_mJ_m2", "beta_110_001_mJ_m2", "beta_110_1m10_mJ_m2",
        "beta_110_1m12_mJ_m2", "beta_110_1m1m1_mJ_m2", "gamma_xy_110_1m12_mJ_m2",
    )
    combination_summary, combination_covariance = _distribution_statistics(combination_rows, quantities)
    bootstrap_summary, bootstrap_covariance = _distribution_statistics(bootstrap_rows, quantities)
    for row in combination_summary:
        row["method"] = "one_replica_per_orientation_combinations"
    for row in bootstrap_summary:
        row["method"] = "replica_block_bootstrap"
    return {
        "primary_orientations": list(selected),
        "pooled": pooled,
        "replica_combinations": combination_rows,
        "replica_block_bootstrap": bootstrap_rows,
        "uncertainty_summary": combination_summary + bootstrap_summary,
        "covariance": {
            "one_replica_per_orientation_combinations": combination_covariance,
            "replica_block_bootstrap": bootstrap_covariance,
        },
    }
