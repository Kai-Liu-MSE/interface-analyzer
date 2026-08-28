# Interface Analyzer v2

`interface-analyzer` sits between an atomistic solid--liquid MD trajectory and
the interfacial quantities one wants to compare with theory: directional
stiffnesses and, when several crystallographic directions are available, cubic
anisotropy parameters. It is intended for equilibrium coexistence simulations
saved as LAMMPS dump files.

The package does **not** replace molecular dynamics. LAMMPS generates the
coexistence trajectory; this package turns that trajectory into an interface
height field, measures its thermal fluctuations, and carries the resulting
replica-to-replica uncertainty into the final fit.

```text
LAMMPS coexistence simulation
        |
        v
raw dump frames  -->  atom-wise structural descriptor  -->  interface h(r_parallel, t)
                                                               |
                                                               v
                                                        native-grid CFM spectrum
                                                               |
                                                               v
                                                   directional stiffness beta(n, t_hat)
                                                               |
                                      independent MD replicas / crystallographic directions
                                                               |
                                                               v
                                          gamma0, epsilon1, epsilon2 + replica uncertainty
```

The runnable [reproducibility bundle](reproducibility/README.md) makes this
chain concrete with a LAMMPS input file, immutable test dumps, post-processing
scripts, and known-answer checks.

## Why capillary fluctuations?

At equilibrium, a solid--liquid interface is not perfectly flat. Thermal
motion makes the local interface height fluctuate along the plane of the
interface. After representing the two interfaces in a periodic coexistence
cell as height fields, their Fourier amplitudes provide a capillary-fluctuation
spectrum. In the long-wavelength regime, the mean squared amplitude is
controlled by the interfacial stiffness: larger stiffness suppresses a mode's
fluctuation.

For every physical mode on the saved uniform grid, v2 forms the CFM response

\[
  \frac{k_B T}{A\langle |h(\mathbf{k})|^2\rangle},
\]

and fits it through the origin to the appropriate quadratic form in
\(\mathbf{k}\). Here \(A\) is the interfacial area and the average includes
the two interfaces and all selected frames. The result is a *stiffness*, not
automatically an interfacial free energy. The two coincide only under the
relevant isotropic limiting assumptions.

This distinction matters for anisotropic interfaces. A stiffness measured in
one in-plane direction is one observable; it cannot by itself determine the
three conventional cubic parameters \(\gamma_0\), \(\epsilon_1\), and
\(\epsilon_2\). v2 therefore keeps the directional stiffnesses explicit and
uses three primary directions—`100_010`, `110_001`, and `110_1m10`—for the
cubic fit.

## Where each part of the package belongs

| Part | Role in the workflow | Main output |
|---|---|---|
| `reproducibility/lammps_inputs/` | Documents how a particular coexistence model was generated. | LAMMPS input deck |
| `analyze_orientation_interface()` | Builds a structural field from each dump frame and locates the upper and lower interfaces. | `h_upper`, `h_lower` |
| `extract_trajectory()` | Applies that extraction consistently to a sequence of frames and records the settings. | Compressed, step-keyed interface pickle |
| `cfm_spectrum()` / `fit_cfm_tensor()` | Converts the height trajectory into physical Fourier modes and estimates directional or tensor stiffness. | Mode spectrum and CFM fit |
| `fit_cubic_replica_blocks()` | Combines replica-resolved directional stiffnesses into a cubic parameter fit. | `gamma0`, `epsilon1`, `epsilon2`, uncertainty distribution |
| `reproducibility/scripts/` | Provides command-line implementations of the three data-reduction stages. | Auditable CSV and JSON files |
| `tests/` | Guards data identity, numerical behavior, and the complete test workflow. | Self-contained regression tests |

The default interface descriptor is orientation-aware local order
(`orientation_phi`), followed by compact kernel coarse graining and the Brown
maximum-difference interface locator. `full_y`, `y_window`, and `2d` modes
cover one-dimensional full-width, finite-width, and two-dimensional interface
representations. Optional PTM-RMSD and Steinhardt / Lechner--Dellago
descriptors are available when a different atom-wise structural signal is
needed.

## Replica uncertainty is part of the result

A Fourier spectrum contains many modes, but the modes from one MD trajectory
are not independent trajectories. v2 treats the complete MD replica as the
statistical unit. Given one stiffness per replica in each primary direction,
it reports:

- the point estimate from the replica mean in each direction;
- every cross-direction one-replica combination (27 fits for three replicas
  in each of three directions); and
- a seeded non-parametric bootstrap that resamples whole replicas within each
  direction, including parameter covariance and percentile intervals.

This is the uncertainty propagated into the cubic parameters. It does not
claim to replace physical convergence checks: temperature, fit window, cell
size, sampling length, descriptor, and interface definition must be selected
and held consistent before replicas are pooled.

## What is included for reproduction

The bundled Al `100[010]` case contains twenty consecutive LAMMPS dump frames,
their SHA-256 manifest, and the input deck from which the model was generated.
It is deliberately small: it verifies parsing, interface extraction,
native-grid Fourier analysis, and output provenance in a few seconds. It is
**not** a converged stiffness calculation and cannot supply the three
directions needed for a physical cubic fit. The companion
`cubic_stiffness_replicas.csv` is a synthetic known-answer fixture solely for
testing the replica-propagation code.

A production study should supply sufficiently long, equilibrated trajectories
and independent replicas for every selected orientation. The full data and
analysis contract are described in the
[reproducibility README](reproducibility/README.md).

## Quick start

Install into an environment with the declared dependencies (including OVITO
Python), or run directly from a checkout:

```bash
python -m pip install .

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

The commands never modify bundled data. They write the interface pickle,
mode-level CFM table, stiffness summary, replica combinations, bootstrap
distribution, covariance, and cubic parameters under the output directories
you provide.

For custom trajectories, use the package CLI to create the interface pickle:

```bash
interface-analyzer-extract path/to/CFGfiles result.pkl.gz \
  --mode 2d --xz-grid 2.5 --y-grid 2.5 --radius 6.0 --window-n 5 \
  --miller-x '0 0 1' --miller-y '1 -1 0' --miller-z '1 1 0'
```

## v2 and the archived v1

v2 is the maintained default branch and uses a compact, native-grid workflow:
no PCHIP interpolation is inserted between the saved interface grid and its
Fourier modes. The original package remains unchanged at
[`legacy/v1`](https://github.com/Kai-Liu-MSE/interface-analyzer/tree/legacy/v1)
for older projects. Its CSP and distance-based LOP utilities are intentionally
not carried forward into v2.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src python -m pytest -q
```

The public suite is self-contained once dependencies are installed. It checks
the immutable raw dump bundle, structural descriptors, CFM behavior, cubic
inversion, replica-block uncertainty propagation, and the bundled end-to-end
workflow.

## Contributors

This package was developed by **Kai Liu (IMDEA)** and **Spearot Douglas
(University of Florida)** under the supervision of
**[Damien Tourret (IMDEA)](https://materiales.imdea.org/people/damien-tourret/)**.
We also acknowledge the contributions of **Qian W**.
