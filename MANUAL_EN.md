# Interface Analyzer Manual (English)

`interface-analyzer` is a Python package for:

- detecting solid-liquid interfaces in Molecular Dynamics (MD) simulations
- performing Capillary Fluctuation Method (CFM) post-processing

## 1. Overview

Main capabilities:

- Interface detection on the `x-z` plane using:
  - `CSP` (Centrosymmetry Parameter)
  - `PTM` (Polyhedral Template Matching)
  - distance-based `LOP`
  - orientation-vector matching `OrientationPhi`
- Brown maximize-difference method to extract `h_upper(x)` and `h_lower(x)`
- CFM spectral analysis and linear fitting utilities
- Custom extension via `PhaseModifierBase`

## 2. Installation

```bash
pip install .
```

Recommended Python version: `>=3.8`.

Dependencies (managed by `setup.py`):

- `numpy`
- `scipy`
- `matplotlib`
- `ovito`
- `tqdm`

## 3. Public API

Main package: `interface_analyzer`

### 3.1 Interface detection functions

- `analyze_by_CSP(...)`
- `analyze_by_PTM(...)`
- `analyze_by_LOP(...)`
- `analyze_by_OrientationPhi(...)`
- `analyze_by_custom_modifier(...)`
- `LOP_analysis(...)` (grid-based distance variant)
- `Orientation_analysis(...)` (grid-based orientation variant)

### 3.2 CFM analysis functions

- `analyze_cfm(...)`
- `plot_cfm_k2_single(...)`
- `analyze_cfm_fit_sensitivity(...)`

### 3.3 Extensible classes/helpers

- `PhaseModifierBase`
- `CSPModifier`
- `PTMModifier`
- `LOPModifier`
- `OrientationPhiModifier`
- `get_orientation_matrix(...)`

## 4. Two LOP-like approaches

This package contains two conceptually different local-order approaches.

### 4.1 Distance-based LOP (no crystal orientation required)

Typical entry points:

- `analyze_by_LOP`
- `LOP_analysis`

Idea:

- Compare neighbor distances to ideal bond length `r_fcc`
- No explicit crystal orientation input is required

Use when:

- orientation is unknown, varying, or not the target descriptor

### 4.2 Orientation-vector matching (orientation required)

Typical entry points:

- `analyze_by_OrientationPhi`
- `Orientation_analysis`

Idea:

- Build rotated FCC reference vectors from Miller directions
- Match actual neighbor vectors to this oriented reference set

Required inputs:

- `miller_x`, `miller_y`, `miller_z`

Constraint:

- the three Miller directions must be pairwise orthogonal

## 5. Recommended workflow

### 5.1 Single-frame interface detection

```python
from interface_analyzer import analyze_by_PTM

res = analyze_by_PTM(
    cfg_path="cfg.100000",
    binsx=150,
    binsz=300,
    n=15,
    rmsd_max=0.10,
)

print(res.keys())
# dict_keys(['phase', 'M', 'x', 'z', 'h_upper', 'h_lower', 'cell'])
```

### 5.2 Batch process frames and save `pkl`

See `Process.py` for a parallel example.  
Expected aggregated format:

```python
results_all[step_id] = {
    "phase": ...,
    "M": ...,
    "x": ...,
    "z": ...,
    "h_upper": ...,
    "h_lower": ...,
    "cell": ...,
}
```

Then save:

```python
import pickle
with open("cfg_post.pkl", "wb") as f:
    pickle.dump(results_all, f)
```

### 5.3 CFM analysis

```python
from interface_analyzer import analyze_cfm

cfm = analyze_cfm(
    pickle_path="cfg_post.pkl",
    T=933.0,         # K
    a=4.05,          # currently stored in output metadata
    use_pchip=True,
    pchipres=1000,
    show_plot=True,
)
```

Returned keys include:

- `k`, `k2`
- `Smax_mean`, `Smin_mean`
- `Ak_max`, `Ak_min`
- `Lx`, `Ly`, `Lz`

### 5.4 Single-file `k^2` linear fit

```python
from interface_analyzer import plot_cfm_k2_single

fit_res = plot_cfm_k2_single(
    filename="cfm_k2_data.txt",
    k2_min=4.0e-4,
    min_points=5,
    a_lattice=4.05,
    L_min_interface=10,
    through_origin=False,
)
print(fit_res)
```

Input file format (default): three columns

1. `k2`
2. upper-interface quantity
3. lower-interface quantity

The function fits the average of columns 2 and 3.

### 5.5 Fit sensitivity analysis

```python
from interface_analyzer import analyze_cfm_fit_sensitivity

sense = analyze_cfm_fit_sensitivity(
    filename="cfm_k2_data.txt",
    k2_min=4.0e-4,
    min_points=5,
    a_lattice=4.05,
    L_min_interface=10,
    through_origin=False,
)
```

## 6. Key parameter guidance

- `binsx`, `binsz`: grid resolution
- `n`: Brown window half-width; often chosen around `box_size/20` in your workflow
- `rmsd_max`: PTM strictness (smaller is stricter)
- `r_fcc`: ideal neighbor distance for distance-based LOP
- `d`: smoothing radius (small: noisy, large: over-smoothed)
- `pchipres`: interpolation resolution for CFM (higher is smoother but slower)

## 7. Custom modifier extension

To extend this package:

1. Inherit `PhaseModifierBase`
2. Implement:
   - `apply_modifier(self, node)`
   - `get_property_name(self)`

Then run through:

```python
from interface_analyzer import analyze_by_custom_modifier

res = analyze_by_custom_modifier("cfg.100000", custom_modifier_instance=my_modifier)
```

## 8. Standard output fields

Interface functions return a dictionary with:

- `M`: `(binsz, binsx)` order-parameter field
- `x`: x-grid centers
- `z`: z-grid centers
- `h_upper`: upper interface profile
- `h_lower`: lower interface profile
- `phase`: phase map (default solid `1`, liquid `2`)
- `cell`: simulation cell matrix

## 9. Citation note

If you use this package in publications, describe:

- OVITO-based structural analysis (CSP/PTM)
- Brown maximize-difference interface extraction
- CFM Fourier-spectrum-based stiffness fitting
