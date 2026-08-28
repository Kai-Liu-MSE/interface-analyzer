from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from interface_analyzer import CUBIC_COEFFICIENTS, fit_cubic_replica_blocks, fit_cubic_stiffness


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "reproducibility" / "data" / "cubic_stiffness_replicas.csv"
FIT_SCRIPT = PROJECT_ROOT / "reproducibility" / "scripts" / "fit_cubic_parameters.py"


def _records():
    with FIXTURE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_cubic_inversion_recovers_the_known_synthetic_parameters():
    parameters = np.array((103.0, 5.0, -0.28))
    stiffnesses = {orientation: float(coefficients @ parameters) for orientation, coefficients in CUBIC_COEFFICIENTS.items()}
    result = fit_cubic_stiffness(stiffnesses)
    assert result["gamma0_mJ_m2"] == pytest.approx(103.0)
    assert result["epsilon1"] == pytest.approx(5.0 / 103.0)
    assert result["epsilon2"] == pytest.approx(-0.28 / 103.0)
    assert result["beta_110_1m12_mJ_m2"] == pytest.approx((2.0 * stiffnesses["110_001"] + stiffnesses["110_1m10"]) / 3.0)


def test_replica_block_propagation_uses_complete_replicas_and_is_reproducible():
    first = fit_cubic_replica_blocks(_records(), bootstrap=64, seed=103)
    second = fit_cubic_replica_blocks(_records(), bootstrap=64, seed=103)
    assert first["pooled"]["gamma0_mJ_m2"] == pytest.approx(103.0)
    assert first["pooled"]["epsilon1"] == pytest.approx(5.0 / 103.0)
    assert first["pooled"]["epsilon2"] == pytest.approx(-0.28 / 103.0)
    assert len(first["replica_combinations"]) == 27
    assert len(first["replica_block_bootstrap"]) == 64
    assert first["replica_block_bootstrap"] == second["replica_block_bootstrap"]
    covariance = np.asarray(first["covariance"]["replica_block_bootstrap"]["matrix"], dtype=float)
    assert covariance.shape == (10, 10)
    assert covariance[0, 0] > 0.0
    row = first["replica_block_bootstrap"][0]
    for orientation in ("100_010", "110_001", "110_1m10"):
        selected = row[f"resampled_replicas_{orientation}"].split(";")
        assert len(selected) == 3
        assert set(selected) <= {"rep1", "rep2", "rep3"}


def test_replica_block_input_rejects_duplicate_or_missing_direction():
    duplicate = _records()
    duplicate.append(dict(duplicate[0]))
    with pytest.raises(ValueError, match="duplicate replica"):
        fit_cubic_replica_blocks(duplicate, bootstrap=1)
    incomplete = [row for row in _records() if row["orientation"] != "110_1m10"]
    with pytest.raises(ValueError, match="No replicas supplied"):
        fit_cubic_replica_blocks(incomplete, bootstrap=1)


def test_cubic_fit_command_writes_an_auditable_replica_distribution(tmp_path):
    environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    completed = subprocess.run(
        [
            sys.executable, str(FIT_SCRIPT), str(FIXTURE), "--output-dir", str(tmp_path),
            "--bootstrap", "16", "--seed", "103",
        ],
        check=True, text=True, capture_output=True, env=environment,
    )
    result = json.loads(completed.stdout)
    assert result["pooled"]["gamma0_mJ_m2"] == pytest.approx(103.0)
    assert (tmp_path / "replica_combinations.csv").is_file()
    assert (tmp_path / "replica_block_bootstrap.csv").is_file()
    assert (tmp_path / "uncertainty_summary.csv").is_file()
    assert (tmp_path / "cubic_interface_parameters.json").is_file()
