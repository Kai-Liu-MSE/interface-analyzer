# 100[010] LAMMPS dump subset

This directory contains 20 contiguous production snapshots from the legacy
100[010] Al solid-liquid coexistence workflow.  Files retain their original
names and bytes.  They span LAMMPS steps 1,000,000 through 1,009,500 at a
uniform 500-step interval.

Run `../../scripts/verify_100_010_dataset.py` from the repository root to
verify every file against `manifest.csv`.  The source repository and immutable
source commit are documented in `../../README.md`.

`manifest.csv` has four columns: filename, timestep, uncompressed byte count,
and SHA-256.  It is intentionally small and text-based so the data identity can
be checked without OVITO or Python package installation.
