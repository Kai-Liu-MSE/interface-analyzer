# Reproducibility Assets

This directory collects the simulation inputs, post-processing scripts, and
analysis notebooks used to reproduce the CFM solid-liquid interface workflow
described in the accompanying manuscript.

The files are organized as a curated workflow rather than a full dump of every
exploratory notebook. Repeated-run notebooks and checkpoint copies were not
copied here unless they serve a distinct purpose.

## Directory Layout

- `lammps_inputs/`: LAMMPS input files for the Al solid-liquid coexistence
  simulations used in the stiffness and anisotropy examples.
- `postprocessing/`: Python scripts for converting production `CFGfiles/`
  snapshots into pickled interface-height datasets.
- `notebooks/`: Jupyter notebooks for thermo-log inspection, stiffness fitting,
  relaxation-time diagnostics, convergence tests, uncertainty propagation, and
  anisotropy fitting.

## Suggested Workflow

1. Run a LAMMPS input from `lammps_inputs/`.
   Each input writes production snapshots to `CFGfiles/` using a name such as
   `cfg.Al_100_010.*`.

2. Convert snapshots to interface-height pickles.
   Use the scripts in `postprocessing/` from the directory containing
   `CFGfiles/`. The output pickle is a dictionary keyed by frame number. Each
   frame contains at least `x`, `h_upper`, `h_lower`, `M`, `phase`, and `cell`.

3. Run CFM stiffness analysis.
   The notebooks call `interface_analyzer.analyze_cfm()` and
   `interface_analyzer.plot_cfm_k2_single()` to compute fluctuation spectra and
   fit stiffness from the low-`k^2` regime.

4. Diagnose sampling and fitting choices.
   Use the relaxation-time, time-convergence, and smoothing/k-window notebooks
   before accepting a stiffness value.

5. Propagate stiffnesses to anisotropy parameters.
   Use the anisotropy notebooks after stiffnesses have been obtained for at
   least three independent orientation/direction combinations.

## Notebook Data Paths

The notebooks are configured to run from a fresh checkout without local absolute
paths. Each notebook defines:

- `DATASET_DIR`: the small bundled CFG dataset for quick local tests.
- `LOCAL_OUTPUT_DIR`: a git-ignored scratch directory under
  `reproducibility/_local_outputs/`.
- `FULL_DATA_ROOT`: the root directory for manuscript-scale generated files.

By default, `FULL_DATA_ROOT` points to `LOCAL_OUTPUT_DIR`, because the full
post-processed datasets are too large to bundle with the repository. After
running the LAMMPS scripts and post-processing the resulting CFG files, point
the notebooks to those generated files with:

```bash
export INTERFACE_ANALYZER_DATA=/path/to/generated/cfm_data
```

For notebooks or cells that process raw CFG snapshots directly, the default
input directory is the bundled `dataset/`. To use a full LAMMPS production
directory instead:

```bash
export INTERFACE_ANALYZER_CFG_DIR=/path/to/CFGfiles
```

## Local Smoke Test

The bundled `dataset/` directory contains a small set of `cfg.Al_100_010.*`
files that can be used to test the Python and notebook workflow without
rerunning LAMMPS.

From the repository root:

```bash
conda activate ace
python interface_analyzer/reproducibility/postprocessing/process_orientation_analysis.py \
  --limit 2 \
  --max_workers 1 \
  --output /tmp/interface_analyzer_smoke/orientation.pkl
```

For PTM/bin averaging:

```bash
python interface_analyzer/reproducibility/postprocessing/process_ptm_bin.py \
  --grid_size 5.0 \
  --limit 1 \
  --max_workers 1 \
  --output /tmp/interface_analyzer_smoke/ptm.pkl
```

The notebook `notebooks/00_local_dataset_smoke_test.ipynb` performs the same
kind of check from Jupyter: it locates the repository root, reads one bundled
CFG file, writes to the git-ignored `reproducibility/_local_outputs/`
directory, and runs `analyze_cfm()`.

## LAMMPS Inputs

- `in.Al_100_010`: base 100[010] model.
- `in.Al_110_001`: 110[001] model.
- `in.Al_110_1-10`: 110[1-10] model.
- `in.Al_110_1-12`: additional 110[1-12] validation orientation.
- `in.Al_111_1-21`: additional 111[1-21] validation orientation.
- `in.Al_110_001_thick`: thickness-sensitivity model.

The inputs reference `../AlSi.aeam`; adjust this path if the potential file is
stored elsewhere.

## Post-Processing Scripts

- `process_orientation_analysis.py`: fixed-parameter orientation-aware LOP
  post-processing for the 100[010] case.
- `process_lop_sweep.py`: command-line sweep over `a_grid` and smoothing radius
  `d` using `Orientation_analysis`.
- `process_ptm_bin.py`: command-line PTM bin-averaging workflow with automatic
  bin counts from the simulation cell.
- `process_ptm_gridsize_fixed.py`: fixed-parameter PTM grid-size workflow.

The legacy `Process_ptm_smoothing_analysis.py` script was not copied because it
imports `PTM_analysis`, which is not currently part of the public
`interface_analyzer` API.

## Notebook Index

- `00_thermo_log_diagnostics.ipynb`: parse and plot LAMMPS thermo-log sections.
- `01_single_orientation_stiffness_100_010.ipynb`: single-orientation
  post-processing and stiffness workflow. This was copied from the old
  `ACE_Al_100_010.ipynb` name and renamed to avoid tying the workflow to a
  particular potential label.
- `02_ptm_bin_average_100_010.ipynb`: PTM/bin-average comparison for 100[010].
- `03_sampling_convergence_100_010.ipynb`: frame-count convergence for 100[010].
- `04_sampling_convergence_110_001.ipynb`: frame-count convergence for 110[001].
- `05_sampling_convergence_110_1-10.ipynb`: frame-count convergence for
  110[1-10].
- `06_sampling_convergence_110_1-12.ipynb`: stiffness workflow for the additional
  110[1-12] orientation.
- `07_sampling_convergence_111_1-21.ipynb`: stiffness workflow for the additional
  111[1-21] orientation.
- `08_relaxation_time_100_010.ipynb`: relaxation-time fit for 100[010].
- `09_relaxation_time_summary.ipynb`: summary plots of mode relaxation times.
- `10_lop_grid_size_and_smoothing_sensitivity.ipynb`: LOP grid-size and
  smoothing-radius sensitivity.
- `11_ptm_grid_size_and_smoothing_sensitivity.ipynb`: PTM grid-size and
  smoothing-radius sensitivity.
- `12_independent_runs_uncertainty.ipynb`: independent-run uncertainty and
  combination statistics.
- `13_time_convergence.ipynb`: unified time-convergence analysis.
- `14_anisotropy_from_orientation_stiffness.ipynb`: solve anisotropy constants
  from orientation-resolved stiffnesses.
- `15_anisotropy_replicate_distribution.ipynb`: propagate replicate variability
  into anisotropy parameters.
- `16_model_thickness_effect.ipynb`: effect of model thickness on anisotropy
  fitting.

## Data Availability Note

The repository contains LAMMPS input scripts and a small CFG sample dataset for
fast local testing. The full manuscript-scale post-processed data products are
not bundled because they are large; they can be regenerated from the LAMMPS
inputs and then used by setting `INTERFACE_ANALYZER_DATA`.
