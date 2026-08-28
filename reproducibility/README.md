# Reproducibility bundle: Al 100[010] interface smoke test

This directory is a small, executable reproduction bundle for
`interface-analyzer` v2.  It follows the curated layout of the legacy public
repository: a documented source dataset, the LAMMPS input that generated the
model, deterministic post-processing scripts, replica-aware cubic fitting,
and machine-checkable reference results.

The bundle is a **software regression and workflow smoke test**.  Twenty
closely spaced frames are enough to check parsing, orientation-aware interface
extraction, native-grid FFT construction, and output provenance.  They are not
long enough to establish a statistically converged interfacial stiffness.

## Layout

- `data/100_010_20frames/`: 20 unmodified LAMMPS custom-dump frames for the
  Al 100[010] solid-liquid coexistence model, plus a byte-level manifest.
- `lammps_inputs/in.Al_100_010`: the source coexistence input deck.
- `scripts/verify_100_010_dataset.py`: verifies filename, timestep, byte size,
  SHA-256, and LAMMPS-dump header before processing.
- `scripts/run_100_010_full_y.py`: performs full-Y orientation extraction and
  writes a compressed interface-height pickle with all scientific parameters
  fixed explicitly.
- `scripts/fit_cfm_stiffness.py`: turns an interface-height pickle into an
  auditable native-grid Fourier-mode table and a zero-intercept CFM stiffness
  fit.
- `scripts/fit_cubic_parameters.py`: turns a replica-resolved directional
  stiffness table into `gamma0`, `epsilon1`, and `epsilon2`, including all
  cross-replica combinations, a block bootstrap, confidence intervals, and
  covariance matrices.
- `data/cubic_stiffness_replicas.csv`: a synthetic, known-answer fixture for
  testing the cubic fit and replica uncertainty propagation.  It is not a
  physical result from the bundled raw frames.
- `expected_results/`: reference summary used by the end-to-end script.
- `notebooks/00_100_010_smoke_test.ipynb`: an interactive entry point for the
  same workflow.

## Provenance and scope

The frames and the LAMMPS input are copied without content changes from the
legacy `interface-analyzer` public repository at commit
[`5d598f5fd80f1689a16bb9a0c1ff770790490c11`](https://github.com/Kai-Liu-MSE/interface-analyzer/commit/5d598f5fd80f1689a16bb9a0c1ff770790490c11):

- `interface_analyzer/reproducibility/dataset/cfg.Al_100_010.1000000` through
  `cfg.Al_100_010.1009500`, inclusive, at 500-step intervals;
- `interface_analyzer/reproducibility/lammps_inputs/in.Al_100_010`.

Despite their `cfg.*` filenames, the bundled snapshots are ordinary textual
LAMMPS custom dumps.  The first frame contains 71,064 atoms.  The manifest is
the authority for content identity; the source commit is recorded here so a
future maintainer can re-fetch or audit the original data.

`lammps_inputs/manifest.csv` records the byte size and SHA-256 of the bundled
input deck as a separate identity check.

The LAMMPS input references an `AlSi.aeam` potential one directory above the
run directory.  The potential is intentionally not bundled here: it is not
needed to reproduce the v2 post-processing result, and rerunning molecular
dynamics requires a separately obtained, explicitly documented potential
artifact.

## Quick start

Install the package into an environment with its declared dependencies, or run
from a checkout with `PYTHONPATH=src`.

```bash
cd interface_analyzer_v2
PYTHONPATH=$PWD/src python reproducibility/scripts/verify_100_010_dataset.py
PYTHONPATH=$PWD/src python reproducibility/scripts/run_100_010_full_y.py \
  --output-dir /tmp/interface_analyzer_v2_100_010/extraction \
  --check-reference
PYTHONPATH=$PWD/src python reproducibility/scripts/fit_cfm_stiffness.py \
  /tmp/interface_analyzer_v2_100_010/extraction/100_010_v2_full_y.pkl.gz \
  --temperature 927 --output-dir /tmp/interface_analyzer_v2_100_010/stiffness
PYTHONPATH=$PWD/src python reproducibility/scripts/fit_cubic_parameters.py \
  reproducibility/data/cubic_stiffness_replicas.csv \
  --bootstrap 1000 --seed 20260828 --output-dir /tmp/interface_analyzer_v2_100_010/cubic
```

The extraction and CFM commands write only to the output directory supplied by
the user.  They never modify bundled data.  The 20-frame stiffness is a
diagnostic output only; do not quote it as a converged physical result.

## Replica uncertainty policy

The cubic fit accepts one CFM stiffness estimate per completed MD replica.
Its point estimate first averages replicas independently within `100_010`,
`110_001`, and `110_1m10`, then solves the established three-parameter cubic
model.  Uncertainty is reported in two complementary ways:

- all cross-orientation one-replica combinations (for three replicas in each
  direction, 3 x 3 x 3 = 27 parameter fits); and
- a seeded non-parametric bootstrap that resamples entire replicas with
  replacement inside each direction.

It is invalid to resample individual Fourier modes from a single MD trajectory
as if they were separate replicas.  Keep temperature, CFM estimator, fit
window, and post-processing settings identical across all rows in a production
stiffness CSV.

## Fixed extraction definition

The script deliberately passes every parameter instead of relying on defaults:

| Setting | Value |
|---|---:|
| Mode | `full_y` |
| Atom-wise descriptor | orientation-aware LOP (`orientation_phi`) |
| Lattice constant | 4.134 Å |
| Box axes X/Y/Z | [100] / [010] / [001] |
| XZ grid target | 2.5 Å |
| Kernel radius | 6.0 Å |
| Brown window | 5 bins |
| CFM temperature for smoke summary | 927 K |
| CFM fit window | `0.005 < k² < 0.03 Å⁻²` |

These settings originate from the legacy 100[010] processing script and are
also written into every v2 output frame under `params`.
