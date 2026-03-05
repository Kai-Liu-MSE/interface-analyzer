# Interface Analyzer

Tools for solid-liquid interface detection in Molecular Dynamics (MD) simulations and Capillary Fluctuation Method (CFM) post-processing.

## Full Manual

- English manual: [MANUAL_EN.md](MANUAL_EN.md)
- Chinese manual: [MANUAL.md](MANUAL.md)

## Features

- Interface detection in `x-z` plane:
  - CSP (`analyze_by_CSP`)
  - PTM (`analyze_by_PTM`)
  - Distance-based LOP (`analyze_by_LOP`, `LOP_analysis`)
  - Orientation-vector matching (`analyze_by_OrientationPhi`, `Orientation_analysis`)
- Brown maximize-difference interface extraction (`h_upper(x)`, `h_lower(x)`)
- CFM spectrum analysis and fitting:
  - `analyze_cfm`
  - `plot_cfm_k2_single`
  - `analyze_cfm_fit_sensitivity`
- Extensible modifier architecture via `PhaseModifierBase`

## Installation

```bash
pip install .
```

Python `>=3.8` is recommended.

## Quick Start

```python
from interface_analyzer import analyze_by_PTM

res = analyze_by_PTM("cfg.100000", binsx=150, binsz=300, n=15, rmsd_max=0.10)
print(res.keys())
```

For complete workflow (batch processing, pickle aggregation, CFM analysis), see [MANUAL.md](MANUAL.md).
