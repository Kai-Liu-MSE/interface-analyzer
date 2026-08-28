"""Steinhardt and Lechner-Dellago atom-wise bond-order descriptors.

The public :func:`compute_bond_order` function accepts either an OVITO
``DataCollection`` or raw positions plus a simulation cell. Neighbor search and
periodic images are delegated to OVITO's cutoff neighbor finder. The returned
``qbar`` quantities follow Lechner-Dellago exactly: complex ``q_lm`` vectors
are averaged before their rotational invariant is evaluated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from ovito.data import CutoffNeighborFinder, DataCollection
try:
    from scipy.special import sph_harm_y
except ImportError:  # SciPy < 1.15
    sph_harm_y = None
    from scipy.special import sph_harm


def _normalize_cell(cell: np.ndarray) -> np.ndarray:
    array = np.asarray(cell, dtype=float)
    if array.shape == (3,):
        if np.any(array <= 0.0):
            raise ValueError("Cell lengths must be positive")
        matrix = np.zeros((3, 4), dtype=float)
        matrix[:3, :3] = np.diag(array)
        return matrix
    if array.shape == (3, 3):
        matrix = np.zeros((3, 4), dtype=float)
        matrix[:3, :3] = array
        return matrix
    if array.shape == (3, 4):
        return array.copy()
    raise ValueError(f"Expected cell shape (3,), (3, 3), or (3, 4); got {array.shape}")


def _normalize_pbc(pbc: bool | Sequence[bool]) -> tuple[bool, bool, bool]:
    if isinstance(pbc, (bool, np.bool_)):
        return bool(pbc), bool(pbc), bool(pbc)
    if len(pbc) != 3:
        raise ValueError(f"Expected three PBC flags, got {pbc!r}")
    return tuple(bool(value) for value in pbc)  # type: ignore[return-value]


def _data_from_arrays(
    positions: np.ndarray, cell: np.ndarray, pbc: bool | Sequence[bool]
) -> DataCollection:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected positions with shape (n_atoms, 3), got {positions.shape}")
    data = DataCollection()
    data.create_particles().create_property("Position", data=positions)
    data.create_cell(matrix=_normalize_cell(cell), pbc=_normalize_pbc(pbc))
    return data


def _invariant(qlm: np.ndarray, degree: int) -> np.ndarray:
    prefactor = 4.0 * math.pi / (2 * degree + 1)
    return np.sqrt(prefactor * np.sum(np.abs(qlm) ** 2, axis=1)).real


def _qlm_from_bonds(
    centers: np.ndarray, vectors: np.ndarray, n_atoms: int, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a complex q_lm vector for every center in a directed bond list."""
    counts = np.bincount(centers, minlength=n_atoms).astype(np.int64, copy=False)
    qlm = np.zeros((n_atoms, 2 * degree + 1), dtype=np.complex128)
    if vectors.size == 0:
        return qlm, counts

    distances = np.linalg.norm(vectors, axis=1)
    if np.any(distances <= 0.0):
        raise ValueError("Neighbor list contains a zero-length bond")
    polar = np.arccos(np.clip(vectors[:, 2] / distances, -1.0, 1.0))
    azimuth = np.arctan2(vectors[:, 1], vectors[:, 0])
    if sph_harm_y is None:
        # scipy.special.sph_harm(m, l, azimuth, polar) in SciPy <= 1.14.
        harmonics = np.column_stack(
            [sph_harm(order, degree, azimuth, polar) for order in range(-degree, degree + 1)]
        )
    else:
        # scipy.special.sph_harm_y(l, m, polar, azimuth) in newer SciPy.
        harmonics = np.column_stack(
            [sph_harm_y(degree, order, polar, azimuth) for order in range(-degree, degree + 1)]
        )
    np.add.at(qlm, centers, harmonics)
    nonzero = counts > 0
    qlm[nonzero] /= counts[nonzero, np.newaxis]
    return qlm, counts


def _ld_average(qlm: np.ndarray, centers: np.ndarray, neighbors: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Average complex q_lm vectors over each atom and its cutoff neighbors."""
    averaged = qlm.copy()
    np.add.at(averaged, centers, qlm[neighbors])
    averaged /= (counts + 1)[:, np.newaxis]
    return averaged


def compute_bond_order(
    positions: np.ndarray | DataCollection,
    cell: np.ndarray | None = None,
    *,
    pbc: bool | Sequence[bool] = True,
    cutoff: float,
    degrees: Sequence[int] = (4, 6),
    averaged: bool = True,
    return_vectors: bool = False,
) -> dict[str, np.ndarray]:
    """Compute raw Steinhardt and Lechner-Dellago bond-order invariants.

    Parameters
    ----------
    positions, cell, pbc
        Raw atom positions and cell, or an OVITO :class:`DataCollection` as
        ``positions`` with ``cell=None``. The raw-array form is convenient for
        diagnostics; the OVITO form reuses the trajectory parser and PBC-aware
        neighbor search used by the rest of the package.
    cutoff
        Explicit radial neighbor cutoff in Angstrom. It is deliberately not
        inferred or normalized by this function.
    degrees
        Harmonic degrees to calculate. ``(4, 6)`` produces ``q4`` and ``q6``.
    averaged
        Also calculate ``qbar_l`` using Lechner-Dellago complex-vector
        averaging over the same cutoff neighbor graph.
    return_vectors
        Include complex ``q{l}m`` and ``qbar{l}m`` arrays for diagnostics.

    Returns
    -------
    dict[str, ndarray]
        Scalar arrays named ``q4``, ``q6``, ``qbar4`` and ``qbar6`` (as
        requested by ``degrees`` and ``averaged``), plus ``neighbor_count``.
    """
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    unique_degrees = tuple(dict.fromkeys(int(degree) for degree in degrees))
    if not unique_degrees or any(degree < 0 for degree in unique_degrees):
        raise ValueError(f"degrees must contain non-negative integers, got {degrees!r}")

    if isinstance(positions, DataCollection):
        if cell is not None:
            raise ValueError("cell must be omitted when passing an OVITO DataCollection")
        data = positions
    else:
        if cell is None:
            raise ValueError("cell is required when passing raw positions")
        data = _data_from_arrays(positions, cell, pbc)

    if data.particles is None:
        raise ValueError("OVITO DataCollection does not contain particles")
    n_atoms = data.particles.count
    pairs, vectors = CutoffNeighborFinder(cutoff, data).find_all()
    centers = np.asarray(pairs[:, 0], dtype=np.intp)
    neighbors = np.asarray(pairs[:, 1], dtype=np.intp)
    vectors = np.asarray(vectors, dtype=float)

    result: dict[str, np.ndarray] = {}
    counts: np.ndarray | None = None
    for degree in unique_degrees:
        qlm, degree_counts = _qlm_from_bonds(centers, vectors, n_atoms, degree)
        if counts is None:
            counts = degree_counts
        result[f"q{degree}"] = _invariant(qlm, degree)
        if return_vectors:
            result[f"q{degree}m"] = qlm
        if averaged:
            qbarlm = _ld_average(qlm, centers, neighbors, degree_counts)
            result[f"qbar{degree}"] = _invariant(qbarlm, degree)
            if return_vectors:
                result[f"qbar{degree}m"] = qbarlm
    result["neighbor_count"] = np.zeros(n_atoms, dtype=np.int64) if counts is None else counts
    return result
