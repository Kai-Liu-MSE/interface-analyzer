from __future__ import annotations

import numpy as np
from ovito.data import CutoffNeighborFinder, DataCollection

from interface_analyzer import compute_bond_order


def fcc_supercell(repeats: int = 3, lattice_constant: float = 4.05) -> tuple[np.ndarray, np.ndarray]:
    basis = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    positions = np.vstack(
        [(basis + (ix, iy, iz)) * lattice_constant for ix in range(repeats) for iy in range(repeats) for iz in range(repeats)]
    )
    return positions, np.eye(3) * (repeats * lattice_constant)


def as_data(positions: np.ndarray, cell: np.ndarray) -> DataCollection:
    data = DataCollection()
    data.create_particles().create_property("Position", data=positions)
    data.create_cell(matrix=np.column_stack((cell, np.zeros(3))), pbc=(True, True, True))
    return data


def test_perfect_fcc_has_12_neighbors_and_known_q_values():
    positions, cell = fcc_supercell()
    result = compute_bond_order(positions, cell, cutoff=3.2)

    assert np.all(result["neighbor_count"] == 12)
    np.testing.assert_allclose(result["q4"], 0.1909406539564933, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result["q6"], 0.5745242597140698, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result["qbar4"], result["q4"], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result["qbar6"], result["q6"], rtol=0.0, atol=1.0e-12)


def test_invariants_survive_rotation_translation_and_permutation():
    positions, cell = fcc_supercell()
    baseline = compute_bond_order(positions, cell, cutoff=3.2)

    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = 0.731
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    rotation = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross
    rotated = compute_bond_order(positions @ rotation.T, rotation @ cell, cutoff=3.2)

    lengths = np.diag(cell)
    translated = np.mod(positions + np.array([1.7, -2.1, 3.3]), lengths)
    permutation = np.random.default_rng(4).permutation(len(positions))
    inverse = np.argsort(permutation)
    permuted = compute_bond_order(translated[permutation], cell, cutoff=3.2)

    for key in ("q4", "q6", "qbar4", "qbar6"):
        np.testing.assert_allclose(rotated[key], baseline[key], rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(permuted[key][inverse], baseline[key], rtol=0.0, atol=1.0e-12)


def test_ld_averages_complex_qlm_not_scalar_q6():
    positions, cell = fcc_supercell()
    positions = positions.copy()
    positions[0] += np.array([0.18, -0.11, 0.07])
    data = as_data(positions, cell)
    result = compute_bond_order(data, cutoff=3.2, degrees=(6,), return_vectors=True)

    pairs, _ = CutoffNeighborFinder(3.2, data).find_all()
    centers, neighbors = pairs[:, 0], pairs[:, 1]
    counts = np.bincount(centers, minlength=len(positions))
    manual_qbar6m = result["q6m"].copy()
    np.add.at(manual_qbar6m, centers, result["q6m"][neighbors])
    manual_qbar6m /= (counts + 1)[:, np.newaxis]
    manual_qbar6 = np.sqrt(4.0 * np.pi / 13.0 * np.sum(np.abs(manual_qbar6m) ** 2, axis=1))

    scalar_average = result["q6"].copy()
    np.add.at(scalar_average, centers, result["q6"][neighbors])
    scalar_average /= counts + 1

    np.testing.assert_allclose(result["qbar6m"], manual_qbar6m, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result["qbar6"], manual_qbar6, rtol=0.0, atol=1.0e-12)
    assert np.max(np.abs(result["qbar6"] - scalar_average)) > 1.0e-5
