# interface-analyzer v2

`interface-analyzer` is a reproducible workflow for solid--liquid interface
analysis from LAMMPS trajectories:

1. generate a coexistence trajectory from a documented LAMMPS input;
2. extract orientation-aware interface heights from raw dump frames;
3. calculate native-grid capillary-fluctuation stiffnesses; and
4. fit cubic interface parameters (`gamma0`, `epsilon1`, `epsilon2`) while
   propagating uncertainty from complete MD replicas.

The runnable [reproducibility bundle](reproducibility/README.md) includes a
small raw LAMMPS-dump regression case, the source input deck, all processing
scripts, and expected numerical checks.  The 20 bundled frames are deliberately
labelled as a software smoke test, not as a converged physical measurement.

Version 2 is the maintained default workflow.  The original implementation is
preserved separately as `legacy/v1`; its CSP and distance-based LOP utilities
are intentionally not part of this release.

## Core design

- `analyze_orientation_interface()` supports `full_y`, `y_window`, and `2d`.
- Atom-wise orientation phi is calculated once per configuration.
- The default result stores only coordinates, interface heights, cell, and
  provenance. The coarse-grained field `M` and phase labels are opt-in via
  `save_field=True`.
- `extract_trajectory()` writes one compressed pickle keyed by LAMMPS step.
- `cfm_spectrum()` and `fit_cfm_tensor()` use the native uniform interface
  grid. They deliberately do not interpolate with PCHIP.
- `fit_cubic_replica_blocks()` converts replica-resolved directional
  stiffnesses into cubic parameters and reports both all cross-replica
  combinations and a complete-replica block bootstrap.  It never resamples
  individual Fourier modes as though they were independent trajectories.
- `ptm_rmsd` reproduces the established PTM scalar: FCC RMSD, otherwise
  `1.5*rmsd_cutoff`; it can use the same 3D compact kernel as other descriptors.
- `compute_bond_order()` provides raw Steinhardt `q4/q6` and correctly
  Lechner-Dellago averaged `qbar4/qbar6` descriptors using an explicit cutoff.

The current reference behavior is the ULux production implementation in
`aeam_sorted/scripts/postprocess/`. The regression tests compare this package
against that implementation for both full-Y and 2D extraction.

## Quick start

```bash
PYTHONPATH=$PWD/src python reproducibility/scripts/verify_100_010_dataset.py
PYTHONPATH=$PWD/src python reproducibility/scripts/run_100_010_full_y.py \
  --output-dir /tmp/interface_analyzer_v2/extraction --check-reference
PYTHONPATH=$PWD/src python reproducibility/scripts/fit_cfm_stiffness.py \
  /tmp/interface_analyzer_v2/extraction/100_010_v2_full_y.pkl.gz \
  --temperature 927 --output-dir /tmp/interface_analyzer_v2/stiffness
PYTHONPATH=$PWD/src python reproducibility/scripts/fit_cubic_parameters.py \
  reproducibility/data/cubic_stiffness_replicas.csv \
  --bootstrap 1000 --seed 20260828 --output-dir /tmp/interface_analyzer_v2/cubic
```

The final command uses a synthetic validation fixture because the raw test
data contains only one orientation.  Replace that CSV with production
replica-resolved stiffnesses for a physical three-orientation fit.

## Local development

```bash
cd /mnt/d/Codex_HPC/ULux/interface_analyzer_v2
PYTHONPATH=$PWD/src conda run --no-capture-output -n ace python scripts/run_regression.py
```

The dependency-free regression runner uses the small 110[001] diagnostic frame
already stored in this ULux workspace. Set `CFM_REGRESSION_CFG` to test another
configuration. The `tests/` directory also contains pytest-formatted versions
for a future CI environment.

## CLI

```bash
interface-analyzer-extract path/to/CFGfiles result.pkl.gz \
  --mode 2d --xz-grid 2.5 --y-grid 2.5 --radius 6.0 --window-n 5 \
  --miller-x '0 0 1' --miller-y '1 -1 0' --miller-z '1 1 0'
```

For a full-Y 1D interface, use `--mode full-y`; for a finite-y-window test,
use `--mode y-window --y-width 8`.

## Optional Steinhardt / Lechner-Dellago q6 descriptor

The bond-order implementation keeps the descriptor layer separate from the
existing kernel smoothing and Brown interface locator. It uses OVITO's directed
cutoff neighbor graph and averages the **complex** `q_lm` vectors before
constructing `qbar_l`; it does not average scalar `q_l` values.

```python
from interface_analyzer import compute_bond_order

result = compute_bond_order(
    positions, cell, pbc=True,
    cutoff=3.82, degrees=(4, 6), averaged=True,
)
q6 = result["q6"]
qbar6 = result["qbar6"]
```

To drive the same interface pipeline with a raw `q6` or Lechner-Dellago
`qbar6` scalar, select the descriptor explicitly. The cutoff is stored in the
compact interface output provenance.

```bash
interface-analyzer-extract path/to/CFGfiles qbar6_interface.pkl.gz \
  --descriptor qbar6 --bond-order-cutoff 3.82 \
  --mode 2d --xz-grid 2.5 --y-grid 2.5 --radius 6.0 --window-n 5
```

`3.82 Å` is only the current explicit CLI default; the production cutoff should
be selected once from the appropriate bulk-Al RDF minimum and then held fixed
for every reference and interface trajectory in a comparison.

## Test suite

The ordinary test suite is self-contained once the declared Python
dependencies are installed:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src python -m pytest -q
```

It tests descriptor invariants, native-grid CFM behavior, immutable bundled
dump data, cubic inversion, and replica-block uncertainty propagation.  The
repository may also contain ignored local validation material used during
development; it is deliberately not required for a public clone.
