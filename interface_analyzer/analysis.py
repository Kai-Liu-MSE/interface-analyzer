import numpy as np
import scipy.fft as spfft
from scipy.fft import rfft, rfftfreq
import scipy.interpolate as spint
import math
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from ovito.io import import_file, export_file 
from ovito.data import NearestNeighborFinder

# --- Updated Imports ---
# Ensure OrientationPhiModifier and the matrix helper are imported from your modifiers file
from .modifiers import (
    _calculate_interfaces, _classify_phase, 
    CSPModifier, PTMModifier, LOPModifier, OrientationPhiModifier,
    PhaseModifierBase, get_orientation_matrix
)

# --- Interface Analysis Wrappers ---

def analyze_by_OrientationPhi(cfg_path, binsx=150, binsz=None, n=None, 
                              lattice_constant=4.05, 
                              miller_x=[1, 1, 0], miller_y=[0, 0, 1], miller_z=[1, -1, 0]):
    """
    Analyzes solid-liquid interfaces using Structural Matching with specific crystal orientation.
    """
    modifier = OrientationPhiModifier(
        binsx=binsx,
        binsz=binsz,
        n=n,
        lattice_constant=lattice_constant,
        miller_x=miller_x,
        miller_y=miller_y,
        miller_z=miller_z
    )
    return modifier.run(cfg_path)
    
def analyze_by_CSP(cfg_path, binsx=150, binsz=None, n=None, nn=12):
    """
    Analyzes solid-liquid interfaces in an atomic configuration using Centrosymmetry Parameter (CSP).
    Uses the CSPModifier class for modular phase identification.

    Parameters:
    - cfg_path (str): Path to the LAMMPS/OVITO configuration file.
    - binsx (int): Number of bins along the X direction.
    - binsz (int, optional): Number of bins along the Z direction.
    - n (int, optional): Half width of the window for the Brown interface method.
    - nn (int): Number of neighbors for the CentroSymmetryModifier.

    Returns:
    - dict: Contains 'phase' matrix, 'M', 'x', 'z', interface heights ('h_upper', 'h_lower'), and 'cell' data.
    """
    # Instantiate the new CSPModifier class and run
    modifier = CSPModifier(
        binsx=binsx,
        binsz=binsz,
        n=n,
        nn=nn
    )
    return modifier.run(cfg_path)


def analyze_by_PTM(cfg_path, binsx=150, binsz=None, n=None, rmsd_max=0.10):
    """
    Analyzes solid-liquid interfaces in an atomic configuration using Polyhedral Template Matching (PTM).
    Uses the PTMModifier class for modular phase identification.

    Parameters:
    - cfg_path (str): Path to the LAMMPS/OVITO configuration file.
    - binsx (int): Number of bins along the X direction.
    - binsz (int, optional): Number of bins along the Z direction.
    - n (int, optional): Half width of the window for the Brown interface method.
    - rmsd_max (float): RMSD cutoff for PTM matching.

    Returns:
    - dict: Contains 'phase' matrix, 'M', 'x', 'z', interface heights ('h_upper', 'h_lower'), and 'cell' data.
    """
    # Instantiate the new PTMModifier class and run
    modifier = PTMModifier(
        binsx=binsx,
        binsz=binsz,
        n=n,
        rmsd_max=rmsd_max
    )
    return modifier.run(cfg_path)
    
def analyze_by_LOP(cfg_path, binsx=150, binsz=None, n=None, r_fcc=2.85, d=10.0, nn_for_phi=12):
    """
    Analyzes solid-liquid interfaces in an atomic configuration using the 
    custom Local Order Parameter (LOP) Phi.
    
    Parameters:
    - cfg_path (str): Path to the LAMMPS/OVITO configuration file.
    - binsx (int): Number of bins along the X direction.
    - binsz (int, optional): Number of bins along the Z direction.
    - n (int, optional): Half width of the window for the Brown interface method.
    - r_fcc (float): Ideal bond length for local distortion (phi) calculation.
    - d (float): Smoothing cylinder radius for weighted average (Phi) calculation.
    - nn_for_phi (int): Number of nearest neighbors for phi calculation.

    Returns:
    - dict: Contains 'phase' matrix, 'M', 'x', 'z', interface heights ('h_upper', 'h_lower'), and 'cell' data.
    """
    # Instantiate the LOPModifier class and run
    modifier = LOPModifier(
        binsx=binsx,
        binsz=binsz,
        n=n,
        r_fcc=r_fcc,
        d=d,
        nn_for_phi=nn_for_phi
    )
    return modifier.run(cfg_path)

def analyze_by_custom_modifier(cfg_path, custom_modifier_instance: PhaseModifierBase):
    """
    Analyzes solid-liquid interfaces using a custom PhaseModifierBase instance.

    This function allows users to plug in their own phase identification logic,
    provided it inherits from PhaseModifierBase and implements the required methods.

    Parameters:
    - cfg_path (str): Path to the LAMMPS/OVITO configuration file.
    - custom_modifier_instance (PhaseModifierBase): An initialized instance of a
      class derived from PhaseModifierBase.

    Returns:
    - dict: Analysis results from the custom modifier's run method.
    """
    if not isinstance(custom_modifier_instance, PhaseModifierBase):
        raise TypeError("custom_modifier_instance must be an instance of a class derived from PhaseModifierBase.")
    
    return custom_modifier_instance.run(cfg_path)

# --- CFM Analysis and Plotting Functions (Unchanged) ---


def analyze_cfm(
    pickle_path,
    T,
    a,
    use_pchip=True,
    pchipres=1000,
    kb=1.380648e-20,
    show_plot=True,
    k_ref=0.2,              # for k^{-2} reference scaling (Å^-1)
):
    """
    Capillary Fluctuation Method (CFM) analysis for a solid-liquid coexistence system.

    It accumulates the ensemble-averaged squared Fourier amplitudes <|A(k)|^2>
    of the interface height fluctuations h(x), and then computes:

        Y(k) = kB*T / (Lx*Ly*<|A(k)|^2>)  ~  γ_tilde * k^2

    Inputs expected in the pickle per snapshot:
        - "cell": (3,3) cell matrix in Å
        - "x":    x-bin centers in Å (length = binsx)
        - "h_upper": upper interface height vs x (Å)
        - "h_lower": lower interface height vs x (Å)
    """

    pickle_path = Path(pickle_path)

    # ----- Load results -----
    with open(pickle_path, "rb") as f:
        results_ptm_all = pickle.load(f)

    # ===== Basic info from the first snapshot =====
    steps = sorted(results_ptm_all.keys())
    first_step = steps[0]

    cell0 = np.array(results_ptm_all[first_step]["cell"])
    Lx = float(cell0[0, 0])
    Ly = float(cell0[1, 1])
    Lz = float(cell0[2, 2])

    x0 = np.array(results_ptm_all[first_step]["x"])
    binsx = len(x0)

    print(f"Lx, Ly, Lz = {Lx:.2f}, {Ly:.2f}, {Lz:.2f} (Å)")
    print(f"Number of x bins: {binsx}")
    print(f"Number of snapshots: {len(steps)}")

    # ===== Build FFT grid (must be periodic: endpoint=False) =====
    if use_pchip:
        npoints = int(pchipres)
        print(f"Using PCHIP interpolation with pchipres = {npoints}")

        # x0 are bin centers; construct a periodic grid covering exactly one period [x_start, x_start+Lx)
        dx_bin = Lx / binsx
        x_start = x0.min() - 0.5 * dx_bin
        x_fft = x_start + np.linspace(0.0, Lx, num=npoints, endpoint=False)
    else:
        npoints = binsx
        print("Using original bins for analysis (assumed periodic bins).")
        x_fft = x0.copy()

    # ===== rFFT frequency axis (only non-negative k) =====
    # rfft returns length nfreq = npoints//2 + 1
    nfreq = npoints // 2 + 1
    k = 2.0 * np.pi * rfftfreq(npoints, d=Lx / npoints)  # Å^-1
    k2 = k**2

    # Accumulate power spectra over snapshots:
    # Smin/Smax store sum over snapshots of |A(k)|^2
    Smin = np.zeros(nfreq, dtype=float)
    Smax = np.zeros(nfreq, dtype=float)

    # ===== Loop over snapshots =====
    for istep in steps:
        data_dict = results_ptm_all[istep]
        x = np.array(data_dict["x"], dtype=float)
        h_upper = np.array(data_dict["h_upper"], dtype=float)
        h_lower = np.array(data_dict["h_lower"], dtype=float)

        # --- Map h(x) onto the periodic FFT grid ---
        if use_pchip:
            # Enforce periodic interpolation by appending the first point at x+Lx
            x_per = np.r_[x, x[0] + Lx]
            hU_per = np.r_[h_upper, h_upper[0]]
            hL_per = np.r_[h_lower, h_lower[0]]

            h1 = spint.pchip_interpolate(x_per, hU_per, x_fft)
            h2 = spint.pchip_interpolate(x_per, hL_per, x_fft)
        else:
            # If you use raw bins, you should ensure x/h are already periodic & equally spaced.
            h1 = h_upper.copy()
            h2 = h_lower.copy()

        # --- Remove the mean (drop k=0 contribution + reduce leakage) ---
        h1 -= np.mean(h1)
        h2 -= np.mean(h2)

        # --- Fourier amplitudes A(k) with correct normalization ---
        # norm="forward" gives A(k) = (1/N) Σ h(x) exp(-ikx), matching CFM convention.
        A1 = rfft(h1, norm="forward")
        A2 = rfft(h2, norm="forward")

        Smax += (np.abs(A1) ** 2)  # upper
        Smin += (np.abs(A2) ** 2)  # lower

    snapshots = len(steps)
    Smax_mean = Smax / snapshots
    Smin_mean = Smin / snapshots

    # ===== Inverted spectrum used for stiffness fitting =====
    # Y(k) = kB*T / (Lx*Ly*<|A(k)|^2>)
    # Note: k=0 is not used in fitting.
    with np.errstate(divide="ignore", invalid="ignore"):
        Ak_max = (kb * T) / (Lx * Ly * Smax_mean)  # units: (mJ/K)*K / Å^2 / Å^2 = mJ/Å^4 (up to your unit system)
        Ak_min = (kb * T) / (Lx * Ly * Smin_mean)

    # ===== Construct k^{-2} reference line (for visual guide only) =====
    # We scale it so that it matches <|A(k)|^2> at k ≈ k_ref for the upper interface
    with np.errstate(divide="ignore", invalid="ignore"):
        ref = k**(-2)

    # choose index near k_ref but avoid k=0
    valid = np.where(k > 0)[0]
    if valid.size > 0:
        kindex = valid[np.argmin(np.abs(k[valid] - k_ref))]
        if np.isfinite(ref[kindex]) and ref[kindex] != 0 and np.isfinite(Smax_mean[kindex]):
            Ak_fit = (Smax_mean[kindex] / ref[kindex]) * ref
        else:
            Ak_fit = np.zeros_like(k)
    else:
        Ak_fit = np.zeros_like(k)

    # ===== Plot =====
    if show_plot:
        plt.figure(figsize=(8, 6))
        plt.loglog(k[1:], Smax_mean[1:], 'o', color='black', label='Upper interface', markersize=6)
        plt.loglog(k[1:], Smin_mean[1:], 'D', color='gray',  label='Lower interface', markersize=6)
        plt.loglog(k[1:], Ak_fit[1:], '-', color='red',
                   label=rf'$k^{{-2}}$ reference (scaled at $k \approx {k_ref}\ \mathrm{{\AA}}^{{-1}}$)')

        plt.xlabel(r'$k$ ($\mathrm{\AA}^{-1}$)')
        plt.ylabel(r'$\langle |A(k)|^2 \rangle$ ($\mathrm{\AA}^2$)')
        plt.legend(loc="lower left", fontsize='small')
        plt.xlim([1e-2, 2e0])
        plt.ylim([1e-6, 1e2])  # you can tune this for your system
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.title('Capillary Fluctuation Method (CFM)')
        plt.tight_layout()
        plt.show()

    results = {
        "k": k,
        "k2": k2,
        "Smin": Smin,
        "Smax": Smax,
        "Smin_mean": Smin_mean,
        "Smax_mean": Smax_mean,
        "Ak_min": Ak_min,   # = kB*T/(Lx*Ly*<|A|^2>)
        "Ak_max": Ak_max,
        "Lx": Lx,
        "Ly": Ly,
        "Lz": Lz,
        "snapshots": snapshots,
        "use_pchip": use_pchip,
        "pchipres": pchipres,
        "T": T,
        "a": a,
    }
    return results

def plot_cfm_k2_single(filename,
                       label=None,
                       k2_min=4.0e-4,
                       min_points=5,
                       a_lattice=4.05,
                       L_min_interface=10,
                       xlim=None,
                       ylim=None,
                       through_origin=False):
    """
    Reads CFM k^2 data from a file and performs a linear fit to determine the
    best fitting range based on maximizing the R^2 value.
    [Immersive content redacted for brevity.]
    """

    # ---------- Read & pre-process ----------
    data = np.loadtxt(filename, comments='#')
    k2 = data[:, 0]
    # Average the Ak values from the two interfaces
    ave = 0.5 * (data[:, 1] + data[:, 2])

    # Sort by k2 (should already be sorted, but ensures robustness)
    order = np.argsort(k2)
    k2 = k2[order]
    ave = ave[order]

    # Apply lower k^2 cutoff
    mask = k2 >= k2_min
    k2 = k2[mask]
    ave = ave[mask]

    # Apply high k^2 cutoff based on min interface width
    # k2_abs_max is the maximum k^2 value that should be physical (high-k noise removed)
    k2_abs_max = ((2 * math.pi) / (L_min_interface * a_lattice)) ** 2
    mask = k2 <= k2_abs_max
    k2 = k2[mask]
    ave = ave[mask]

    if len(k2) < min_points:
        raise ValueError("Not enough data points after filtering.")

    # ---------- Find the n that gives max R^2 ----------
    best_r2 = -1e99
    best_m = best_b = None
    best_n = min_points

    N = len(k2)

    # Iterate over all possible subsets of data starting from the low-k side
    for n in range(min_points, N + 1):
        x = k2[:n]
        y = ave[:n]

        # Fit model: y = m x (+ b)
        if through_origin:
            # y = m x
            m = np.linalg.lstsq(x[:, None], y, rcond=None)[0][0]
            b = 0.0
        else:
            # y = m x + b
            m, b = np.polyfit(x, y, 1)

        # Compute R² (Coefficient of determination)
        y_pred = m * x + b
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 1.0

        # Track the best (maximum R²) fit
        if r2 > best_r2:
            best_r2 = r2
            best_m = m
            best_b = b
            best_n = n

    # Extract best fitting range
    k2_used = k2[:best_n]
    ave_used = ave[:best_n]

    # ---------- Plot ----------
    if label is None:
        label = Path(filename).name

    fit_type = "y=mx" if through_origin else "y=mx+b"

    plt.figure(figsize=(8, 6))

    # plot used points
    plt.plot(k2, ave, 'o', color='gray', markersize=4, label='Data (all points)')
    plt.plot(k2_used, ave_used, 'o', color='black', markersize=6, label=f"Points Used for Fit ({best_n})")

    # fitted line
    kfit = np.linspace(k2.min(), k2_used.max() * 1.1, 200)
    yfit = best_m * kfit + best_b
    plt.plot(kfit, yfit, 'r--', label=f"Fit ({fit_type})\nm={best_m:.2e}, b={best_b:.2e}\nR²={best_r2:.3f}")

    plt.xlabel(r'$k^2$ ($\mathrm{\AA}^{-2}$)')
    # Note: Ak is proportional to the slope/stiffness coefficient $\gamma$
    plt.ylabel(r'$k_{B}T / (L_x L_y \langle|A(k)|^2\rangle)$ (mJ/$\mathrm{\AA}^4$)')

    plt.legend(loc="upper left", fontsize='small')

    # Apply limits
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.title(f'CFM $k^2$ Linear Fit: {label}')
    plt.tight_layout()
    plt.show()

    # ---------- Return results ----------
    return {
        "file": str(filename),
        "slope": best_m,
        "intercept": best_b,
        "r2": best_r2,
        "k2_min_used": float(k2_used.min()),
        "k2_max_used": float(k2_used.max()),
        "n_points": best_n,
        "through_origin": through_origin
    }


def analyze_cfm_fit_sensitivity(filename,
                                k2_min=4.0e-4,
                                min_points=5,
                                a_lattice=4.05,
                                L_min_interface=10,
                                through_origin=False):
    """
    Analyzes the sensitivity of the Capillary Fluctuation Method (CFM) linear fit
    (slope/stiffness and R^2) as a function of the number of low-k^2 points used.
    [Immersive content redacted for brevity.]
    """

    # ---------- Read & pre-process ----------
    data = np.loadtxt(filename, comments='#')
    k2 = data[:, 0]
    ave = 0.5 * (data[:, 1] + data[:, 2])

    order = np.argsort(k2)
    k2 = k2[order]
    ave = ave[order]

    # Apply lower k^2 cutoff
    mask = k2 >= k2_min
    k2 = k2[mask]
    ave = ave[mask]

    # Apply high k^2 cutoff
    k2_abs_max = ((2 * math.pi) / (L_min_interface * a_lattice)) ** 2
    mask = k2 <= k2_abs_max
    k2 = k2[mask]
    ave = ave[mask]

    N = len(k2)
    if N < min_points:
        raise ValueError("Not enough data points after initial filtering (N < min_points).")

    # ---------- Iterate and Fit ----------
    n_points_array = []
    stiffness_array = []
    r2_array = []

    # Iterate over all possible subsets of data starting from the low-k side
    for n in range(min_points, N + 1):
        x = k2[:n]
        y = ave[:n]

        # Fit model: y = m x (+ b)
        if through_origin:
            m = np.linalg.lstsq(x[:, None], y, rcond=None)[0][0]
            # b is effectively 0
        else:
            m, b = np.polyfit(x, y, 1)

        # Compute R² (Coefficient of determination)
        # Note: we use the two-parameter fit (y = mx + b) R^2 calculation even if b is forced to 0
        # for a standard comparison.
        y_pred = m * x if through_origin else m * x + b
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 1.0

        # Store results
        n_points_array.append(n)
        stiffness_array.append(m)
        r2_array.append(r2)

    # ---------- Plotting ----------
    fit_label = "y=mx" if through_origin else "y=mx+b"
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Plot 1: Interface Stiffness (Slope m) vs. Number of Points (n)
    axes[0].plot(n_points_array, stiffness_array, 'o-', color='navy', markersize=5)
    axes[0].set_ylabel(r'Interface Stiffness (Slope $m$)', fontsize=12)
    axes[0].set_title(f'CFM Fit Sensitivity ({Path(filename).name}) - Fit Type: {fit_label}', fontsize=14)
    axes[0].grid(True, which='major', ls='--')

    # Plot 2: R^2 vs. Number of Points (n)
    axes[1].plot(n_points_array, r2_array, 's-', color='darkgreen', markersize=5)
    axes[1].set_xlabel('Number of Points Used in Fit ($n$)', fontsize=12)
    axes[1].set_ylabel(r'Coefficient of Determination ($R^2$)', fontsize=12)
    axes[1].set_ylim(max(0, min(r2_array) * 0.95), 1.005) # Ensure R2 scale starts near 0 or min R2
    axes[1].grid(True, which='major', ls='--')

    plt.tight_layout()
    plt.show()

    return {
        "n_points": np.array(n_points_array),
        "stiffness": np.array(stiffness_array),
        "r2": np.array(r2_array),
        "through_origin": through_origin
    }

def LOP_analysis(
    cfg_path,
    a_grid=3.52,          # Grid spacing (Å), typically close to lattice constant
    r_fcc=2.49,           # Ideal FCC nearest-neighbor distance used in φ calculation
    d=7.0,                # Smoothing radius for grid-based averaging (Å)
    nn_for_phi=12,        # Number of nearest neighbors used for φ
    cutoff_factor=1.5,    # Safety margin for neighbor search (not strictly used here)
    n=None,               # Window size for Brown interface-finding method
    solid_value=1,
    liquid_value=2
):
    """
    Compute a grid-based Local Order Parameter (LOP) field in the X–Z plane
    by directly averaging atomic φ values around each grid point.

    This function performs the following steps:
    ---------------------------------------------------------------
    1. Loads a configuration and computes per-atom φ_i values,
       where φ_i measures the local structural deviation from
       ideal FCC neighbors (following Asadi's definition).

    2. Creates a uniform grid in the X–Z plane. The grid spacing
       is controlled by 'a_grid' and is typically chosen close to
       the lattice constant (~3.5 Å for metals).

    3. For each grid point g(x,z), finds nearby atoms within a
       cylindrical radius 'd' and computes a weighted average of
       their φ_i values:
           LOP(g) = Σ w(r_ig) φ_i  /  Σ w(r_ig)
       where w(r) = (1 − (r/d)^2)^2 is the cylinder-weight function.

       This produces a smooth field of LOP values *independent of
       atomic positions*, unlike the original LOPModifier which
       first computes atomic averages and then bins them.

    4. Applies Brown’s window method on the grid-based LOP field
       to determine upper and lower interface height functions
       h_upper(x) and h_lower(x).

    5. Classifies each grid point as solid or liquid depending on
       its position relative to the interface heights.

    Returns:
    ---------------------------------------------------------------
    A dictionary containing:
        M         : (binsz, binsx) LOP values on the grid
        x, z      : grid center coordinates along X and Z
        h_upper   : upper interface height function
        h_lower   : lower interface height function
        phase     : classified phase map (solid / liquid)
        cell      : simulation cell matrix

    Notes:
    ---------------------------------------------------------------
    - This method allows LOP to be evaluated *directly on grid points*
      rather than atoms, avoiding some noise related to atomic binning.

    - Grid spacing (a_grid) and smoothing radius (d) together determine
      the spatial resolution and smoothness of the LOP field.

    - This function integrates naturally into the CFM pipeline, since
      the Brown interface-finding method is applied to the grid field.
    """

    # ----- 1. Load configuration and compute atomic φ values -----
    node = import_file(str(cfg_path))
    data = node.compute()

    positions = data.particles.positions
    num_particles = data.particles.count

    # Extract simulation cell and box dimensions
    cell = np.array(data.cell[:])
    Lx, Ly, Lz = cell[0, 0], cell[1, 1], cell[2, 2]
    COx, COy, COz = cell[0, 3], cell[1, 3], cell[2, 3]

    # Nearest-neighbor finder used to compute φ_i
    finder_phi = NearestNeighborFinder(N=nn_for_phi, data_collection=data)
    neigh_idx, neigh_vec = finder_phi.find_all()
    dists = np.linalg.norm(neigh_vec, axis=2)

    # φ_i = (1/N) Σ (|r_ij| − r_fcc)^2
    phi = np.mean((dists - r_fcc)**2, axis=1)

    # ----- 2. Construct uniform X–Z grid -----
    # Number of grid points determined by desired spacing a_grid
    binsx = max(1, int(round(Lx / a_grid)))
    binsz = max(1, int(round(Lz / a_grid)))

    # Actual grid spacing
    hx = Lx / binsx
    hz = Lz / binsz

    # Grid point centers in physical coordinates
    x_centers = COx + (np.arange(binsx) + 0.5) * hx
    z_centers = COz + (np.arange(binsz) + 0.5) * hz

    # For Brown method: window size automatically determined if not given
    if n is None:
        n = max(5, binsz // 20)

    # ----- 3. Grid-based weighted averaging of atomic φ values -----
    # Each grid point stores:
    #   numerator = Σ w φ_i
    #   denominator = Σ w
    numerator_grid = np.zeros((binsz, binsx), dtype=float)
    denominator_grid = np.zeros((binsz, binsx), dtype=float)

    # Maximum integer grid radius based on smoothing radius d
    Nx_rad = int(np.ceil(d / hx))
    Nz_rad = int(np.ceil(d / hz))

    # Pre-compute fractional coordinates for efficient PBC handling
    fx = ((positions[:, 0] - COx) / Lx) % 1.0
    fz = ((positions[:, 2] - COz) / Lz) % 1.0

    # Loop over atoms and distribute their φ_i to nearby grid points
    for i in range(num_particles):
        phi_i = phi[i]

        # Approximate grid index of this atom in X and Z
        ix = int(fx[i] * binsx)  # 0..binsx-1
        iz = int(fz[i] * binsz)  # 0..binsz-1

        # Loop over neighboring grid points within radius d
        for dx_idx in range(-Nx_rad, Nx_rad + 1):
            ix_g = (ix + dx_idx) % binsx  # periodic boundary

            # Real-space distance in X direction
            xg = x_centers[ix_g]
            dx_real = positions[i, 0] - xg
            if dx_real > 0.5 * Lx: dx_real -= Lx
            elif dx_real < -0.5 * Lx: dx_real += Lx

            for dz_idx in range(-Nz_rad, Nz_rad + 1):
                iz_g = (iz + dz_idx) % binsz

                zg = z_centers[iz_g]
                dz_real = positions[i, 2] - zg
                if dz_real > 0.5 * Lz: dz_real -= Lz
                elif dz_real < -0.5 * Lz: dz_real += Lz

                # Radial distance squared in the X–Z plane
                r_xz_sq = dx_real * dx_real + dz_real * dz_real
                if r_xz_sq > d * d:
                    continue

                r = np.sqrt(r_xz_sq)

                # Cylinder weight function: w(r) = (1 − (r/d)^2)^2
                w = (1.0 - (r / d) ** 2) ** 2

                numerator_grid[iz_g, ix_g] += w * phi_i
                denominator_grid[iz_g, ix_g] += w

    # Compute final grid LOP values
    M = np.zeros_like(numerator_grid)
    mask_nonzero = denominator_grid > 1e-12
    M[mask_nonzero] = numerator_grid[mask_nonzero] / denominator_grid[mask_nonzero]

    # ----- 4. Apply Brown method to extract interface positions -----
    h_upper, h_lower = _calculate_interfaces(M, z_centers, n)

    # ----- 5. Phase classification of grid points -----
    phase = _classify_phase(
        binsx, binsz,
        x_centers, z_centers,
        h_upper, h_lower,
        solid_value=solid_value,
        liquid_value=liquid_value
    )

    # ----- 6. Return full results -----
    results = {
        "M": M,
        "x": x_centers,
        "z": z_centers,
        "h_upper": h_upper,
        "h_lower": h_lower,
        "phase": phase,
        "cell": cell
    }

    return results
    
def Orientation_analysis(
    cfg_path,
    lattice_constant=4.05,
    miller_x=[1, 1, 0], 
    miller_y=[0, 0, 1], 
    miller_z=[1, -1, 0],
    a_grid=3.52,          # Grid spacing (Å)
    d=7.0,                # Smoothing radius for grid-based averaging (Å)
    n=None,               # Window size for Brown interface-finding method
    solid_value=1,
    liquid_value=2
):
    """
    Compute a grid-based Orientation Order Parameter field in the X–Z plane.
    Similar to LOP_analysis, but uses crystal orientation Miller indices for Phi calculation.
    """

    # ----- 1. Load configuration and compute orientation-aware phi values -----
    node = import_file(str(cfg_path))
    data = node.compute()
    positions = data.particles.positions
    num_particles = data.particles.count

    # Extract simulation cell
    cell = np.array(data.cell[:])
    Lx, Ly, Lz = cell[0, 0], cell[1, 1], cell[2, 2]
    COx, COy, COz = cell[0, 3], cell[1, 3], cell[2, 3]

    # Calculate orientation-aware phi (Vectorized logic)
    rotation_matrix = get_orientation_matrix(miller_x, miller_y, miller_z)
    ref_vecs = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ]) * (lattice_constant / 2.0)
    ref_vecs = ref_vecs @ rotation_matrix.T

    finder_phi = NearestNeighborFinder(N=12, data_collection=data)
    _, neigh_vecs = finder_phi.find_all() 

    # Broad-casting comparison: (N, 12_actual, 12_ideal, 3)
    diff = neigh_vecs[:, :, np.newaxis, :] - ref_vecs[np.newaxis, np.newaxis, :, :]
    phi = np.sum(np.min(np.sum(diff**2, axis=3), axis=2), axis=1)

    # ----- 2. Construct uniform X–Z grid -----
    binsx = max(1, int(round(Lx / a_grid)))
    binsz = max(1, int(round(Lz / a_grid)))
    hx, hz = Lx / binsx, Lz / binsz
    x_centers = COx + (np.arange(binsx) + 0.5) * hx
    z_centers = COz + (np.arange(binsz) + 0.5) * hz

    if n is None:
        n = max(5, binsz // 20)

    # ----- 3. Grid-based weighted averaging (using logic from LOP_analysis) -----
    numerator_grid = np.zeros((binsz, binsx), dtype=float)
    denominator_grid = np.zeros((binsz, binsx), dtype=float)

    Nx_rad, Nz_rad = int(np.ceil(d / hx)), int(np.ceil(d / hz))
    fx = ((positions[:, 0] - COx) / Lx) % 1.0
    fz = ((positions[:, 2] - COz) / Lz) % 1.0

    for i in range(num_particles):
        phi_i = phi[i]
        ix, iz = int(fx[i] * binsx), int(fz[i] * binsz)

        for dx_idx in range(-Nx_rad, Nx_rad + 1):
            ix_g = (ix + dx_idx) % binsx
            xg = x_centers[ix_g]
            dx_real = positions[i, 0] - xg
            if dx_real > 0.5 * Lx: dx_real -= Lx
            elif dx_real < -0.5 * Lx: dx_real += Lx

            for dz_idx in range(-Nz_rad, Nz_rad + 1):
                iz_g = (iz + dz_idx) % binsz
                zg = z_centers[iz_g]
                dz_real = positions[i, 2] - zg
                if dz_real > 0.5 * Lz: dz_real -= Lz
                elif dz_real < -0.5 * Lz: dz_real += Lz

                r_xz_sq = dx_real**2 + dz_real**2
                if r_xz_sq > d * d: continue

                w = (1.0 - (np.sqrt(r_xz_sq) / d) ** 2) ** 2
                numerator_grid[iz_g, ix_g] += w * phi_i
                denominator_grid[iz_g, ix_g] += w

    M = np.zeros_like(numerator_grid)
    mask = denominator_grid > 1e-12
    M[mask] = numerator_grid[mask] / denominator_grid[mask]

    # ----- 4. Apply Brown method and classify -----
    h_upper, h_lower = _calculate_interfaces(M, z_centers, n)
    phase = _classify_phase(binsx, binsz, x_centers, z_centers, h_upper, h_lower, solid_value, liquid_value)

    return {"M": M, "x": x_centers, "z": z_centers, "h_upper": h_upper, "h_lower": h_lower, "phase": phase, "cell": cell}
