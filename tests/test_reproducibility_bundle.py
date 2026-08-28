from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "reproducibility" / "data" / "100_010_20frames"
EXTRACT_SCRIPT = PROJECT_ROOT / "reproducibility" / "scripts" / "run_100_010_full_y.py"
STIFFNESS_SCRIPT = PROJECT_ROOT / "reproducibility" / "scripts" / "fit_cfm_stiffness.py"


def test_bundled_100_010_dataset_is_complete_and_byte_identical():
    with (DATA_DIR / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert [int(row["timestep"]) for row in rows] == list(range(1_000_000, 1_009_501, 500))
    for row in rows:
        path = DATA_DIR / row["filename"]
        assert path.stat().st_size == int(row["bytes"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        with path.open(encoding="ascii") as handle:
            assert handle.readline().strip() == "ITEM: TIMESTEP"
            assert int(handle.readline()) == int(row["timestep"])


def test_bundled_lammps_input_declares_the_matching_100_010_dump_cadence():
    input_dir = PROJECT_ROOT / "reproducibility" / "lammps_inputs"
    with (input_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    path = input_dir / row["filename"]
    assert path.stat().st_size == int(row["bytes"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    text = path.read_text(encoding="utf-8")
    assert re.search(r"lattice\s+fcc \$a orient x 1 0 0 orient y 0 1 0 orient z 0 0 1", text)
    assert re.search(r"dump\s+id all custom 500 CFGfiles/cfg\.Al_100_010\.\* id type x y z", text)


def test_bundled_raw_frames_run_through_extraction_and_native_grid_stiffness(tmp_path):
    environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    extraction = tmp_path / "extraction"
    subprocess.run(
        [sys.executable, str(EXTRACT_SCRIPT), "--output-dir", str(extraction), "--workers", "2", "--check-reference"],
        check=True, text=True, capture_output=True, env=environment,
    )
    interface_pickle = extraction / "100_010_v2_full_y.pkl.gz"
    assert interface_pickle.is_file()
    stiffness = tmp_path / "stiffness"
    completed = subprocess.run(
        [
            sys.executable, str(STIFFNESS_SCRIPT), str(interface_pickle), "--temperature", "927",
            "--output-dir", str(stiffness),
        ],
        check=True, text=True, capture_output=True, env=environment,
    )
    summary = json.loads(completed.stdout)
    assert summary["fit"]["status"] == "ok"
    assert summary["fit"]["n_modes"] == 10
    assert (stiffness / "native_grid_cfm_modes.csv").is_file()
    assert (stiffness / "cfm_stiffness.json").is_file()
