"""Native-grid capillary-fluctuation spectra and through-origin fits.

No interpolation is performed: a saved interface grid defines the physical
Fourier modes that may be fitted.
"""

from __future__ import annotations

import gzip
import math
import pickle
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

KB_CFM_NUMERIC = 1.380649  # kB in the Angstrom/mJ CFM convention.


def load_interface_pickle(path: str | Path) -> dict[Any, dict[str, Any]]:
    path = Path(path)
    with path.open("rb") as handle:
        magic = handle.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(path, "rb") as handle:
        result = pickle.load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"Expected a frame dictionary, got {type(result).__name__}")
    return result


def _steps(results: dict[Any, Any]) -> list[Any]:
    try:
        return sorted(results, key=int)
    except (ValueError, TypeError):
        return sorted(results, key=str)


def _geometry(frame: dict[str, Any]) -> tuple[float, float, float, int, int]:
    cell = np.asarray(frame["cell"], dtype=float)
    lx, ly, lz = float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])
    h = np.asarray(frame["h_upper"], dtype=float).squeeze()
    if h.ndim == 1:
        return lx, ly, lz, h.size, 1
    if h.ndim == 2:
        ny, nx = h.shape
        return lx, ly, lz, nx, ny
    raise ValueError(f"Expected h_upper with one or two dimensions, got {h.shape}")


def _height(frame: dict[str, Any], key: str, nx: int, ny: int, lz: float, unwrap_z: bool) -> np.ndarray:
    h = np.asarray(frame[key], dtype=float).squeeze()
    if h.ndim == 1:
        if h.size != nx:
            raise ValueError(f"Unexpected 1D height shape {h.shape}")
        h = h[np.newaxis, :]
    if h.shape != (ny, nx):
        raise ValueError(f"Expected (Ny,Nx)={(ny, nx)}, got {h.shape}")
    if not unwrap_z:
        return h
    phase = 2.0 * np.pi * h / lz
    return np.unwrap(np.unwrap(phase, axis=1), axis=0) * lz / (2.0 * np.pi)


def _signed_indices(n: int) -> np.ndarray:
    return np.rint(np.fft.fftfreq(n) * n).astype(int)


def _independent_modes(lx: float, ly: float, nx: int, ny: int) -> dict[str, np.ndarray]:
    rows: list[tuple[int, int, int, int, float, float]] = []
    for iy, my in enumerate(_signed_indices(ny)):
        for ix, mx in enumerate(_signed_indices(nx)):
            if mx == 0 and my == 0:
                continue
            self_conjugate = ((-ix) % nx == ix) and ((-iy) % ny == iy)
            if not ((mx > 0) or (mx == 0 and my > 0) or self_conjugate):
                continue
            rows.append((ix, iy, int(mx), int(my), 2.0 * np.pi * mx / lx, 2.0 * np.pi * my / ly))
    values = np.asarray(rows, dtype=float)
    k2 = values[:, 4] ** 2 + values[:, 5] ** 2
    theta = np.degrees(np.arctan2(values[:, 5], values[:, 4]))
    # Match the established ULux analysis convention exactly. Equal-k2 modes
    # must have a deterministic angular order for array-level regression.
    order = np.lexsort((theta, k2))
    values = values[order]
    return {
        "ix": values[:, 0].astype(int), "iy": values[:, 1].astype(int),
        "nx": values[:, 2].astype(int), "ny": values[:, 3].astype(int),
        "kx_Ainv": values[:, 4], "ky_Ainv": values[:, 5], "k2_Ainv2": values[:, 4] ** 2 + values[:, 5] ** 2,
        "theta_deg": np.degrees(np.arctan2(values[:, 5], values[:, 4])),
    }


def cfm_spectrum(
    results: dict[Any, dict[str, Any]], temperature: float, *, unwrap_z: bool = True
) -> dict[str, Any]:
    """Return independent native-grid 2D modes and their mean CFM response."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    steps = _steps(results)
    if not steps:
        raise ValueError("No interface frames")
    lx, ly, lz, nx, ny = _geometry(results[steps[0]])
    modes = _independent_modes(lx, ly, nx, ny)
    # Accumulate mode powers online. Retaining one (n_frames, n_modes) array
    # serves no numerical purpose and becomes needlessly expensive for the
    # 10,001-frame production trajectories.
    power_u_sum = np.zeros(len(modes["ix"]), dtype=float)
    power_l_sum = np.zeros(len(modes["ix"]), dtype=float)
    projection_error = 0.0
    for step in steps:
        frame = results[step]
        current = np.asarray(frame["cell"], dtype=float)
        if not np.allclose((current[0, 0], current[1, 1]), (lx, ly), rtol=1.0e-7, atol=1.0e-5):
            raise ValueError("In-plane cell changes require fractional-coordinate remapping")
        hu = _height(frame, "h_upper", nx, ny, float(current[2, 2]), unwrap_z)
        hl = _height(frame, "h_lower", nx, ny, float(current[2, 2]), unwrap_z)
        au, al = np.fft.fft2(hu - hu.mean(), norm="forward"), np.fft.fft2(hl - hl.mean(), norm="forward")
        power_u_sum += np.abs(au[modes["iy"], modes["ix"]]) ** 2
        power_l_sum += np.abs(al[modes["iy"], modes["ix"]]) ** 2
        projected = np.fft.fft(hu.mean(axis=0) - hu.mean(), norm="forward")
        zero_y = modes["ny"] == 0
        if np.any(zero_y):
            projection_error = max(projection_error, float(np.max(np.abs(au[0, modes["ix"][zero_y]] - projected[modes["ix"][zero_y]]))))
    power_upper = power_u_sum / len(steps)
    power_lower = power_l_sum / len(steps)
    power = 0.5 * (power_upper + power_lower)
    with np.errstate(divide="ignore", invalid="ignore"):
        response = KB_CFM_NUMERIC * temperature / (lx * ly * power)
    return {
        **modes, "power_upper_A2": power_upper, "power_lower_A2": power_lower, "power_combined_A2": power,
        "response_mJ_m2_Ainv2": response, "temperature_K": float(temperature), "Lx_A": lx, "Ly_A": ly, "Lz_A": lz,
        "grid_shape_yx": (ny, nx), "steps": steps, "projection_identity_max_abs_difference_A": projection_error,
    }


def fit_cfm_tensor(
    spectrum: dict[str, Any], *, k2_min: float = 0.005, k2_max: float = 0.03,
    model: Literal["ky0", "kx0", "diagonal", "full", "isotropic"] = "full",
) -> dict[str, Any]:
    """Fit a CFM response with a zero-intercept model on native grid modes."""
    kx, ky, k2, response = (np.asarray(spectrum[key], dtype=float) for key in ("kx_Ainv", "ky_Ainv", "k2_Ainv2", "response_mJ_m2_Ainv2"))
    mask = np.isfinite(response) & (k2 > k2_min) & (k2 < k2_max)
    names: list[str]
    if model == "ky0":
        mask &= np.asarray(spectrum["ny"]) == 0
        design, names = kx[mask, None] ** 2, ["gamma_xx_mJ_m2"]
    elif model == "kx0":
        mask &= np.asarray(spectrum["nx"]) == 0
        design, names = ky[mask, None] ** 2, ["gamma_yy_mJ_m2"]
    elif model == "diagonal":
        design, names = np.column_stack((kx[mask] ** 2, ky[mask] ** 2)), ["gamma_xx_mJ_m2", "gamma_yy_mJ_m2"]
    elif model == "full":
        design, names = np.column_stack((kx[mask] ** 2, 2.0 * kx[mask] * ky[mask], ky[mask] ** 2)), ["gamma_xx_mJ_m2", "gamma_xy_mJ_m2", "gamma_yy_mJ_m2"]
    elif model == "isotropic":
        design, names = k2[mask, None], ["gamma_iso_mJ_m2"]
    else:
        raise ValueError(f"Unknown model {model!r}")
    y = response[mask]
    rank = int(np.linalg.matrix_rank(design)) if len(design) else 0
    result: dict[str, Any] = {"model": model, "n_modes": int(len(y)), "rank": rank, "k2_min": k2_min, "k2_max": k2_max}
    if len(y) <= design.shape[1] or rank < design.shape[1]:
        return {**result, "status": "rank_deficient"}
    beta, _, _, singular = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ beta
    residual = y - predicted
    total = np.sum((y - y.mean()) ** 2)
    result.update(status="ok", r2=float(1.0 - np.sum(residual ** 2) / total) if total else float("nan"), rmse=float(np.sqrt(np.mean(residual ** 2))), condition_number=float(np.linalg.cond(design)), singular_values=singular.tolist())
    result.update(dict(zip(names, map(float, beta))))
    return result
