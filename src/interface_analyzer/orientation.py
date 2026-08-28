"""Orientation-aware LOP and compact interface extraction.

The full-Y path reproduces the numerical definition used by the legacy
``Orientation_analysis`` routine and the ULux production postprocessor. The
2D path generalizes the coarse-graining field to a periodic (z, y, x) grid.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from ovito.data import DataCollection, NearestNeighborFinder
from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier

from .bond_order import compute_bond_order

InterfaceMode = Literal["full_y", "y_window", "2d"]
CoarseGraining = Literal["kernel", "bin"]
DescriptorName = Literal["orientation_phi", "ptm_rmsd", "q4", "q6", "qbar4", "qbar6"]


def parse_miller(value: str | Sequence[int]) -> tuple[int, int, int]:
    """Parse ``'1 -1 0'`` or a three-integer sequence."""
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    if len(value) != 3:
        raise ValueError(f"Expected three Miller indices, got {value!r}")
    return tuple(int(component) for component in value)  # type: ignore[return-value]


def orientation_matrix(
    miller_x: Sequence[int], miller_y: Sequence[int], miller_z: Sequence[int]
) -> np.ndarray:
    axes = [np.asarray(axis, dtype=float) for axis in (miller_x, miller_y, miller_z)]
    if any(np.linalg.norm(axis) == 0.0 for axis in axes):
        raise ValueError("Miller axes must be non-zero")
    unit_axes = [axis / np.linalg.norm(axis) for axis in axes]
    dots = [np.dot(unit_axes[0], unit_axes[1]), np.dot(unit_axes[0], unit_axes[2]), np.dot(unit_axes[1], unit_axes[2])]
    if not np.allclose(dots, 0.0, atol=1.0e-5):
        raise ValueError(f"Miller directions are not orthogonal: dot products={dots}")
    return np.asarray(unit_axes, dtype=float)


def compute_orientation_phi(
    data: DataCollection,
    lattice_constant: float,
    miller_x: Sequence[int],
    miller_y: Sequence[int],
    miller_z: Sequence[int],
) -> np.ndarray:
    """Calculate the orientation-aware 12-neighbor local order parameter.

    For every actual nearest-neighbor vector, the closest rotated FCC
    ``a/2 <110>`` reference vector is selected. The 12 resulting squared
    deviations are summed. This is the corrected descriptor used in the
    current ULux production workflow.
    """
    rotation = orientation_matrix(miller_x, miller_y, miller_z)
    reference = np.asarray(
        [
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        ],
        dtype=float,
    ) * (lattice_constant / 2.0)
    reference = reference @ rotation.T
    finder = NearestNeighborFinder(N=12, data_collection=data)
    _, neighbor_vectors = finder.find_all()
    differences = neighbor_vectors[:, :, np.newaxis, :] - reference[np.newaxis, np.newaxis, :, :]
    return np.sum(np.min(np.sum(differences * differences, axis=3), axis=2), axis=1)


def compute_ptm_rmsd_scalar(data: DataCollection, rmsd_cutoff: float) -> np.ndarray:
    """Return the PTM scalar used by the established full-Y PTM workflow.

    FCC-recognized atoms retain their PTM RMSD. Every other local environment
    is assigned ``1.5 * rmsd_cutoff`` so that the scalar remains low in the
    oriented solid and high in liquid/disordered regions.
    """
    if rmsd_cutoff <= 0.0:
        raise ValueError("PTM RMSD cutoff must be positive")
    structure = np.asarray(data.particles["Structure Type"], dtype=int)
    rmsd = np.asarray(data.particles["RMSD"], dtype=float)
    return np.where(structure == 1, rmsd, 1.5 * rmsd_cutoff)


def _cell_components(cell: np.ndarray) -> tuple[float, float, float, float, float, float]:
    if cell.shape[0] < 3 or cell.shape[1] < 4:
        raise ValueError(f"Expected OVITO 3x4 cell, got {cell.shape}")
    off_diagonal = cell[:3, :3] - np.diag(np.diag(cell[:3, :3]))
    if np.max(np.abs(off_diagonal)) > 1.0e-8 * max(float(np.max(np.abs(cell[:3, :3]))), 1.0):
        raise ValueError("Only orthorhombic axis-aligned cells are supported")
    lx, ly, lz = (float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2]))
    ox, oy, oz = (float(cell[0, 3]), float(cell[1, 3]), float(cell[2, 3]))
    if min(lx, ly, lz) <= 0.0:
        raise ValueError(f"Non-positive cell lengths: {(lx, ly, lz)}")
    return lx, ly, lz, ox, oy, oz


def _periodic_difference(values: np.ndarray, centers: np.ndarray, length: float) -> np.ndarray:
    difference = values - centers
    return np.where(difference > 0.5 * length, difference - length, np.where(difference < -0.5 * length, difference + length, difference))


def _brown_interfaces_1d(field_zx: np.ndarray, z: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    bins_z, bins_x = field_zx.shape
    if n < 1 or 2 * n >= bins_z:
        raise ValueError(f"Invalid Brown window n={n} for {bins_z} z bins")
    psi = np.zeros_like(field_zx, dtype=float)
    for iz in range(n, bins_z - n):
        psi[iz] = (np.sum(field_zx[iz + 1 : iz + n + 1], axis=0) - np.sum(field_zx[iz - n : iz], axis=0)) / n
    upper = z[np.argmax(psi, axis=0)]
    lower = z[np.argmin(psi, axis=0)]
    swap = lower > upper
    return np.where(swap, lower, upper), np.where(swap, upper, lower)


def _brown_interfaces_2d(field_zyx: np.ndarray, z: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    bins_z = field_zyx.shape[0]
    if n < 1 or 2 * n >= bins_z:
        raise ValueError(f"Invalid Brown window n={n} for {bins_z} z bins")
    psi = np.zeros_like(field_zyx, dtype=float)
    for iz in range(n, bins_z - n):
        psi[iz] = (np.sum(field_zyx[iz + 1 : iz + n + 1], axis=0) - np.sum(field_zyx[iz - n : iz], axis=0)) / n
    upper = z[np.argmax(psi, axis=0)]
    lower = z[np.argmin(psi, axis=0)]
    swap = lower > upper
    return np.where(swap, lower, upper), np.where(swap, upper, lower)


def _grid_1d(
    positions: np.ndarray,
    phi: np.ndarray,
    cell: np.ndarray,
    grid: float,
    radius: float,
    n: int | None,
    y_width: float | None,
    save_field: bool,
) -> dict[str, object]:
    lx, ly, lz, ox, oy, oz = _cell_components(cell)
    bins_x, bins_z = max(1, int(round(lx / grid))), max(1, int(round(lz / grid)))
    hx, hz = lx / bins_x, lz / bins_z
    x = ox + (np.arange(bins_x) + 0.5) * hx
    z = oz + (np.arange(bins_z) + 0.5) * hz
    window_n = n if n is not None else max(5, bins_z // 20)
    n_slabs = 1 if y_width is None or y_width <= 0.0 else max(1, int(round(ly / y_width)))
    y_edges = oy + np.linspace(0.0, ly, n_slabs + 1)

    fx = ((positions[:, 0] - ox) / lx) % 1.0
    fy = ((positions[:, 1] - oy) / ly) % 1.0
    fz = ((positions[:, 2] - oz) / lz) % 1.0
    y_bin = np.minimum((fy * n_slabs).astype(int), n_slabs - 1)
    x_bin = np.minimum((fx * bins_x).astype(int), bins_x - 1)
    z_bin = np.minimum((fz * bins_z).astype(int), bins_z - 1)
    x_wrapped = ox + fx * lx
    z_wrapped = oz + fz * lz
    nx_radius, nz_radius = int(math.ceil(radius / hx)), int(math.ceil(radius / hz))

    upper_all, lower_all, fields, phases, counts = [], [], [], [], []
    for slab in range(n_slabs):
        selected = np.flatnonzero(y_bin == slab)
        counts.append(int(selected.size))
        numerator = np.zeros((bins_z, bins_x), dtype=float)
        denominator = np.zeros((bins_z, bins_x), dtype=float)
        for dx_index in range(-nx_radius, nx_radius + 1):
            ix = (x_bin[selected] + dx_index) % bins_x
            dx = _periodic_difference(x_wrapped[selected], x[ix], lx)
            for dz_index in range(-nz_radius, nz_radius + 1):
                iz = (z_bin[selected] + dz_index) % bins_z
                dz = _periodic_difference(z_wrapped[selected], z[iz], lz)
                r2 = dx * dx + dz * dz
                mask = r2 <= radius * radius
                if not np.any(mask):
                    continue
                weights = (1.0 - r2[mask] / (radius * radius)) ** 2
                flat = iz[mask] * bins_x + ix[mask]
                np.add.at(numerator.ravel(), flat, weights * phi[selected][mask])
                np.add.at(denominator.ravel(), flat, weights)
        field = np.zeros_like(numerator)
        nonzero = denominator > 1.0e-12
        field[nonzero] = numerator[nonzero] / denominator[nonzero]
        upper, lower = _brown_interfaces_1d(field, z, window_n)
        upper_all.append(upper)
        lower_all.append(lower)
        if save_field:
            fields.append(field.astype(np.float32))
            phase = np.full(field.shape, 2, dtype=np.int16)
            for ix in range(bins_x):
                phase[(z >= lower[ix]) & (z <= upper[ix]), ix] = 1
            phases.append(phase)

    result: dict[str, object] = {
        "x": x.astype(np.float32), "z": z.astype(np.float32), "y_edges": y_edges.astype(np.float32),
        "h_upper": np.asarray(upper_all, dtype=np.float32), "h_lower": np.asarray(lower_all, dtype=np.float32),
        "cell": cell.astype(np.float32), "grid_shape": (bins_z, bins_x), "grid_spacing_actual": (hx, hz),
        "y_width_actual": np.diff(y_edges).astype(np.float32), "slab_atom_counts": np.asarray(counts, dtype=np.int32), "n": int(window_n),
    }
    if save_field:
        result["M"] = np.asarray(fields, dtype=np.float32)
        result["phase"] = np.asarray(phases, dtype=np.int16)
    return result


def _grid_2d(
    positions: np.ndarray, phi: np.ndarray, cell: np.ndarray, xz_grid: float, y_grid: float,
    radius: float, n: int | None, save_field: bool,
) -> dict[str, object]:
    lx, ly, lz, ox, oy, oz = _cell_components(cell)
    bins_x, bins_y, bins_z = max(1, int(round(lx / xz_grid))), max(1, int(round(ly / y_grid))), max(1, int(round(lz / xz_grid)))
    hx, hy, hz = lx / bins_x, ly / bins_y, lz / bins_z
    x = ox + (np.arange(bins_x) + 0.5) * hx
    y = oy + (np.arange(bins_y) + 0.5) * hy
    z = oz + (np.arange(bins_z) + 0.5) * hz
    window_n = n if n is not None else max(5, bins_z // 20)

    fx, fy, fz = ((positions[:, axis] - origin) / length % 1.0 for axis, origin, length in ((0, ox, lx), (1, oy, ly), (2, oz, lz)))
    ix0, iy0, iz0 = np.minimum((fx * bins_x).astype(int), bins_x - 1), np.minimum((fy * bins_y).astype(int), bins_y - 1), np.minimum((fz * bins_z).astype(int), bins_z - 1)
    xw, yw, zw = ox + fx * lx, oy + fy * ly, oz + fz * lz
    numerator, denominator = np.zeros((bins_z, bins_y, bins_x), dtype=float), np.zeros((bins_z, bins_y, bins_x), dtype=float)
    d2 = radius * radius
    for dz_index in range(-int(math.ceil(radius / hz)), int(math.ceil(radius / hz)) + 1):
        iz = (iz0 + dz_index) % bins_z
        dz2 = _periodic_difference(zw, z[iz], lz) ** 2
        for dy_index in range(-int(math.ceil(radius / hy)), int(math.ceil(radius / hy)) + 1):
            iy = (iy0 + dy_index) % bins_y
            dyz2 = dz2 + _periodic_difference(yw, y[iy], ly) ** 2
            for dx_index in range(-int(math.ceil(radius / hx)), int(math.ceil(radius / hx)) + 1):
                ix = (ix0 + dx_index) % bins_x
                r2 = dyz2 + _periodic_difference(xw, x[ix], lx) ** 2
                mask = r2 <= d2
                if not np.any(mask):
                    continue
                weights = (1.0 - r2[mask] / d2) ** 2
                flat = (iz[mask] * bins_y + iy[mask]) * bins_x + ix[mask]
                np.add.at(numerator.ravel(), flat, weights * phi[mask])
                np.add.at(denominator.ravel(), flat, weights)
    field = np.zeros_like(numerator, dtype=np.float32)
    nonzero = denominator > 1.0e-12
    field[nonzero] = (numerator[nonzero] / denominator[nonzero]).astype(np.float32)
    upper, lower = _brown_interfaces_2d(field, z, window_n)
    result: dict[str, object] = {
        "x": x.astype(np.float32), "y": y.astype(np.float32), "z": z.astype(np.float32),
        "h_upper": upper.astype(np.float32), "h_lower": lower.astype(np.float32), "cell": cell.astype(np.float32),
        "grid_shape": (bins_z, bins_y, bins_x), "grid_spacing_actual": (hx, hy, hz), "y_grid_target": float(y_grid),
        "n": int(window_n), "empty_grid_fraction": float(1.0 - np.count_nonzero(nonzero) / nonzero.size),
    }
    if save_field:
        result["M"] = field
        phase = np.full(field.shape, 2, dtype=np.int16)
        z3 = z[:, np.newaxis, np.newaxis]
        phase[(z3 >= lower[np.newaxis]) & (z3 <= upper[np.newaxis])] = 1
        result["phase"] = phase
    return result


def _grid_2d_bin(
    positions: np.ndarray, phi: np.ndarray, cell: np.ndarray, xz_grid: float, y_grid: float,
    n: int | None, save_field: bool,
) -> dict[str, object]:
    """Directly average an atom-wise scalar into periodic 3D grid cells.

    This is the genuine 2D-interface counterpart of the established PTM bin
    workflow: the descriptor is averaged in each (z, y, x) voxel, without a
    smoothing kernel, and Brown's locator is then applied independently at
    every (y, x) location.
    """
    lx, ly, lz, ox, oy, oz = _cell_components(cell)
    bins_x, bins_y, bins_z = (
        max(1, int(round(lx / xz_grid))),
        max(1, int(round(ly / y_grid))),
        max(1, int(round(lz / xz_grid))),
    )
    hx, hy, hz = lx / bins_x, ly / bins_y, lz / bins_z
    x = ox + (np.arange(bins_x) + 0.5) * hx
    y = oy + (np.arange(bins_y) + 0.5) * hy
    z = oz + (np.arange(bins_z) + 0.5) * hz
    window_n = n if n is not None else max(5, bins_z // 20)

    fx, fy, fz = (
        (positions[:, axis] - origin) / length % 1.0
        for axis, origin, length in ((0, ox, lx), (1, oy, ly), (2, oz, lz))
    )
    ix = np.minimum((fx * bins_x).astype(int), bins_x - 1)
    iy = np.minimum((fy * bins_y).astype(int), bins_y - 1)
    iz = np.minimum((fz * bins_z).astype(int), bins_z - 1)
    flat = (iz * bins_y + iy) * bins_x + ix
    sums = np.zeros((bins_z, bins_y, bins_x), dtype=float)
    counts = np.zeros((bins_z, bins_y, bins_x), dtype=np.int32)
    np.add.at(sums.ravel(), flat, phi)
    np.add.at(counts.ravel(), flat, 1)
    nonzero = counts > 0
    field = np.zeros_like(sums, dtype=np.float32)
    field[nonzero] = (sums[nonzero] / counts[nonzero]).astype(np.float32)
    upper, lower = _brown_interfaces_2d(field, z, window_n)
    result: dict[str, object] = {
        "x": x.astype(np.float32), "y": y.astype(np.float32), "z": z.astype(np.float32),
        "h_upper": upper.astype(np.float32), "h_lower": lower.astype(np.float32), "cell": cell.astype(np.float32),
        "grid_shape": (bins_z, bins_y, bins_x), "grid_spacing_actual": (hx, hy, hz), "y_grid_target": float(y_grid),
        "n": int(window_n), "empty_grid_fraction": float(1.0 - np.count_nonzero(nonzero) / nonzero.size),
    }
    if save_field:
        result["M"] = field
        phase = np.full(field.shape, 2, dtype=np.int16)
        z3 = z[:, np.newaxis, np.newaxis]
        phase[(z3 >= lower[np.newaxis]) & (z3 <= upper[np.newaxis])] = 1
        result["phase"] = phase
    return result


def analyze_orientation_interface(
    cfg_path: str | Path, *, mode: InterfaceMode = "full_y", lattice_constant: float = 4.05,
    miller_x: Sequence[int] = (0, 0, 1), miller_y: Sequence[int] = (1, -1, 0), miller_z: Sequence[int] = (1, 1, 0),
    xz_grid: float = 2.5, y_grid: float = 2.5, y_width: float | None = None,
    radius: float = 6.0, n: int | None = 5, save_field: bool = False,
    coarse_graining: CoarseGraining = "kernel",
    descriptor: DescriptorName = "orientation_phi", bond_order_cutoff: float = 3.82,
    ptm_rmsd_cutoff: float = 0.15,
) -> dict[str, object]:
    """Extract an interface from one CFG using a selectable atom-wise descriptor.

    ``orientation_phi`` is the production orientation-aware LOP.
    ``ptm_rmsd`` uses the established PTM RMSD scalar. ``q4``, ``q6``,
    ``qbar4``, and ``qbar6`` are Steinhardt / Lechner-Dellago scalars. All
    descriptors then share the same coarse-graining and Brown locator.
    """
    if xz_grid <= 0.0 or y_grid <= 0.0:
        raise ValueError("Grid spacings must be positive")
    if coarse_graining not in {"kernel", "bin"}:
        raise ValueError(f"Unknown coarse-graining method {coarse_graining!r}")
    if coarse_graining == "kernel" and radius <= 0.0:
        raise ValueError("Kernel radius must be positive")
    if coarse_graining == "bin" and mode != "2d":
        raise ValueError("Direct bin coarse-graining is currently implemented only for mode='2d'")
    pipeline = import_file(str(cfg_path))
    if descriptor == "ptm_rmsd":
        ptm = PolyhedralTemplateMatchingModifier(output_rmsd=True, rmsd_cutoff=ptm_rmsd_cutoff)
        ptm.structures[PolyhedralTemplateMatchingModifier.Type.HCP].enabled = False
        ptm.structures[PolyhedralTemplateMatchingModifier.Type.BCC].enabled = False
        pipeline.modifiers.append(ptm)
    data = pipeline.compute()
    positions, cell = np.asarray(data.particles.positions), np.asarray(data.cell[:])
    if descriptor == "orientation_phi":
        values = compute_orientation_phi(data, lattice_constant, miller_x, miller_y, miller_z)
    elif descriptor == "ptm_rmsd":
        values = compute_ptm_rmsd_scalar(data, ptm_rmsd_cutoff)
    elif descriptor in {"q4", "q6", "qbar4", "qbar6"}:
        values = compute_bond_order(
            data, cutoff=bond_order_cutoff, degrees=(4, 6), averaged=True
        )[descriptor]
    else:
        raise ValueError(f"Unknown atom-wise descriptor {descriptor!r}")
    if mode == "2d":
        if coarse_graining == "kernel":
            result = _grid_2d(positions, values, cell, xz_grid, y_grid, radius, n, save_field)
        else:
            result = _grid_2d_bin(positions, values, cell, xz_grid, y_grid, n, save_field)
    elif mode in {"full_y", "y_window"}:
        result = _grid_1d(positions, values, cell, xz_grid, radius, n, None if mode == "full_y" else y_width, save_field)
        if mode == "full_y":
            # A true 1D interface has the established legacy layout h(x).
            result["h_upper"] = np.asarray(result["h_upper"])[0]
            result["h_lower"] = np.asarray(result["h_lower"])[0]
            if save_field:
                result["M"] = np.asarray(result["M"])[0]
                result["phase"] = np.asarray(result["phase"])[0]
    else:
        raise ValueError(f"Unknown interface mode {mode!r}")
    result["params"] = {
        "mode": mode, "descriptor": descriptor, "lattice_constant": float(lattice_constant), "miller_x": list(miller_x),
        "miller_y": list(miller_y), "miller_z": list(miller_z), "xz_grid": float(xz_grid),
        "y_grid": float(y_grid), "y_width": None if y_width is None else float(y_width), "radius": float(radius),
        "coarse_graining": coarse_graining,
        "n": result["n"], "save_field": bool(save_field),
    }
    if descriptor != "orientation_phi":
        if descriptor == "ptm_rmsd":
            result["params"]["ptm_rmsd_cutoff"] = float(ptm_rmsd_cutoff)
            result["params"]["ptm_scalar_definition"] = "RMSD if FCC else 1.5*rmsd_cutoff"
        else:
            result["params"]["bond_order_cutoff"] = float(bond_order_cutoff)
    return result
