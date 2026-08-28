from __future__ import annotations

import numpy as np

from interface_analyzer import cfm_spectrum, fit_cfm_tensor


def test_native_grid_ky0_spectrum_and_fit():
    lx, ly, lz, nx, ny = 300.0, 20.0, 180.0, 120, 3
    x = (np.arange(nx) + 0.5) * lx / nx
    y = (np.arange(ny) + 0.5) * ly / ny
    profile = sum(0.8 / order * np.sin(2.0 * np.pi * order * x / lx) for order in range(1, 10))
    upper = 130.0 + np.tile(profile, (ny, 1))
    lower = 50.0 + np.tile(profile, (ny, 1))
    frames = {
        step: {"x": x, "y": y, "h_upper": upper, "h_lower": lower, "cell": np.array([[lx, 0, 0, 0], [0, ly, 0, 0], [0, 0, lz, 0]])}
        for step in (0, 1000, 2000)
    }
    spectrum = cfm_spectrum(frames, 925.0)
    assert spectrum["projection_identity_max_abs_difference_A"] < 1e-12
    fit = fit_cfm_tensor(spectrum, model="ky0", k2_min=0.005, k2_max=0.03)
    assert fit["status"] == "ok"
    assert fit["n_modes"] >= 5
