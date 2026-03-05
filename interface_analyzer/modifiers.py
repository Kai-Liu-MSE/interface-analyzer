import numpy as np
from abc import ABC, abstractmethod
from ovito.io import import_file
from ovito.data import NearestNeighborFinder, CutoffNeighborFinder
from ovito.modifiers import (
    CentroSymmetryModifier, PolyhedralTemplateMatchingModifier, ComputePropertyModifier,
    SpatialBinningModifier, PythonScriptModifier
)

# --- Private Helper Functions (used by all concrete modifiers) ---

def _run_binning(node, prop_name, binsx, binsz):
    """
    Calculates the averaged value of prop_name in bins on the XZ plane.
    """
    node.modifiers.append(SpatialBinningModifier(
        property=prop_name,
        direction=SpatialBinningModifier.Direction.XZ,
        bin_count=(binsx, binsz),
        reduction_operation=SpatialBinningModifier.Operation.Mean
    ))
    data = node.compute()
    grid = data.grids['binning']

    M_1d = np.asarray(grid[prop_name])
    M = M_1d.reshape((-1, binsx))  # (binsz, binsx)

    cell = data.cell[:]  # 3x4 matrix
    Lx, Lz = cell[0, 0], cell[2, 2]
    COx, COz = cell[0, 3], cell[2, 3]

    x_centers = COx + (np.arange(binsx) + 0.5) * (Lx / binsx)
    z_centers = COz + (np.arange(binsz) + 0.5) * (Lz / binsz)

    return M, x_centers, z_centers, cell

def _calculate_interfaces(M, z_centers, n):
    """
    Uses the Brown maximize difference method to determine the upper and lower interface bounds.
    """
    binsz, binsx = M.shape
    h_upper = np.zeros(binsx)
    h_lower = np.zeros(binsx)

    for dx in range(binsx):
        psi = np.zeros(binsz)
        # Calculate psi(i) = (Avg[i+1 to i+n] - Avg[i-1 to i-n]) / n
        for i in range(n, binsz - n):
            # Sum from i+1 to i+n
            phipos = np.sum(M[i + 1 : i + n + 1, dx]) 
            # Sum from i-n to i-1
            phineg = np.sum(M[i - n : i, dx])
            psi[i] = (phipos - phineg) / n

        wmax = int(np.argmax(psi))
        wmin = int(np.argmin(psi))
        h_upper[dx] = z_centers[wmax]
        h_lower[dx] = z_centers[wmin]

    swap_mask = h_lower > h_upper
    if np.any(swap_mask):
        tmp = h_lower.copy()
        h_lower[swap_mask] = h_upper[swap_mask]
        h_upper[swap_mask] = tmp[swap_mask]

    return h_upper, h_lower

def _classify_phase(binsx, binsz, x_centers, z_centers, h_upper, h_lower,
                     solid_value=1, liquid_value=2):
    """
    Determines the phase of each bin (solid/liquid) based on the calculated interface bounds.
    """
    phase = np.full((binsz, binsx), liquid_value, dtype=np.int16)
    for ix in range(binsx):
        zu = h_upper[ix]
        zl = h_lower[ix]
        # Only classify particles between the upper and lower interfaces as solid
        mask_z = (z_centers >= zl) & (z_centers <= zu)
        phase[mask_z, ix] = solid_value
    return phase


# --- Abstract Base Class (The interface for extensibility) ---

class PhaseModifierBase(ABC):
    """
    Abstract Base Class for solid-liquid phase identification modifiers.
    """
    def __init__(self, binsx=150, binsz=None, n=None, solid_value=1, liquid_value=2):
        self.binsx = binsx
        self.binsz = binsz if binsz is not None else binsx * 2
        self.n = n if n is not None else max(5, self.binsz // 20)
        self.solid_value = solid_value
        self.liquid_value = liquid_value
        self.results = None

    @abstractmethod
    def apply_modifier(self, node):
        """
        Applies the specific OVITO modifier(s) to the pipeline node.
        """
        pass

    @abstractmethod
    def get_property_name(self):
        """
        Returns the name of the particle property created or used by the modifier.
        """
        pass

    def run(self, cfg_path):
        """
        Runs the entire analysis pipeline: import -> modify -> bin -> find interfaces.
        """
        node = import_file(str(cfg_path))

        # 1. Apply phase identification modifier(s)
        self.apply_modifier(node)
        prop_name = self.get_property_name()

        # 2. Bin the resulting property
        M, xcs, zcs, cell = _run_binning(node, prop_name, self.binsx, self.binsz)

        # 3. Calculate interface height profiles
        h_upper, h_lower = _calculate_interfaces(M, zcs, self.n)

        # 4. Classify phases
        phase = _classify_phase(self.binsx, self.binsz, xcs, zcs, h_upper, h_lower,
                                 self.solid_value, self.liquid_value)

        self.results = dict(phase=phase, M=M, x=xcs, z=zcs, h_upper=h_upper, h_lower=h_lower, cell=cell)
        return self.results

# --- Concrete Implementations (CSP and PTM) ---

class CSPModifier(PhaseModifierBase):
    """Phase identification using the CentroSymmetry Parameter (CSP)."""
    def __init__(self, nn=12, **kwargs):
        super().__init__(**kwargs)
        self.nn = nn

    def apply_modifier(self, node):
        node.modifiers.append(CentroSymmetryModifier(num_neighbors=self.nn))

    def get_property_name(self):
        return 'Centrosymmetry'

class PTMModifier(PhaseModifierBase):
    """Phase identification using Polyhedral Template Matching (PTM)."""
    def __init__(self, rmsd_max=0.10, **kwargs):
        super().__init__(**kwargs)
        self.rmsd_max = rmsd_max
        self.custom_property_name = 'SolidFlag_PTM'

    def apply_modifier(self, node):
        # PTM setup: only look for FCC structure (StructureType==1)
        PTM = PolyhedralTemplateMatchingModifier(output_rmsd=True, rmsd_cutoff=self.rmsd_max)
        PTM.structures[PolyhedralTemplateMatchingModifier.Type.HCP].enabled = False
        PTM.structures[PolyhedralTemplateMatchingModifier.Type.BCC].enabled = False
        node.modifiers.append(PTM)

        # Compute a custom property: RMSD if FCC, high value otherwise (liquid/non-solid).
        expr = f"(StructureType==1) ? RMSD : (1.5*{self.rmsd_max})"
        CPM = ComputePropertyModifier(
            output_property=self.custom_property_name,
            expressions=[expr],
            only_selected=False,
            operate_on='particles'
        )
        node.modifiers.append(CPM)

    def get_property_name(self):
        return self.custom_property_name

# --- Optimized LOP Implementation ---

class LOPModifier(PhaseModifierBase):
    """
    Phase identification using the custom Local Order Parameter (LOP) Phi.
    Fully vectorized implementation using OVITO's neighbor finders.
    """
    def __init__(self, r_fcc=2.85, d=10.0, nn_for_phi=12, **kwargs):
        super().__init__(**kwargs)
        self.r_fcc = r_fcc
        self.d = d
        self.nn_for_phi = nn_for_phi
        self.final_property_name = 'LocalOrderParameter_Phi'

    def _calculate_lop(self, frame, data):
        # 1. Compute Local Distortion (phi) using NearestNeighborFinder
        # Computes N nearest neighbors for every particle.
        # finder.find_all() returns (indices, vectors).
        # indices shape: (N_particles, N_neighbors)
        # vectors shape: (N_particles, N_neighbors, 3)
        finder_phi = NearestNeighborFinder(N=self.nn_for_phi, data_collection=data)
        neigh_idx, neigh_vec = finder_phi.find_all()
        
        # Calculate distances (Euclidean norm along the last axis)
        dists = np.linalg.norm(neigh_vec, axis=2)
        
        # Vectorized calculation of phi for all particles
        # phi = (1/12) * sum((r_i - r_fcc)^2)
        phi = np.sum((dists - self.r_fcc)**2, axis=1) / self.nn_for_phi
        
        # Store phi temporarily (optional, but good for debugging or if needed elsewhere)
        # data.particles_.create_property('LocalDistortion_phi', data=phi)

        # 2. Compute Weighted Average (Phi) using CutoffNeighborFinder
        # Finds all neighbors within distance 'd'.
        # finder.find_all() returns (indices, vectors) in a flattened format.
        # indices shape: (M, 2) where M is total number of neighbor pairs.
        # vectors shape: (M, 3)
        finder_smooth = CutoffNeighborFinder(self.d, data)
        c_neigh_idx, c_neigh_vec = finder_smooth.find_all()
        
        # Extract columns for clarity
        central_indices = c_neigh_idx[:, 0]
        neighbor_indices = c_neigh_idx[:, 1]
        
        # Calculate cylindrical distance (XZ plane)
        # c_neigh_vec is (M, 3) -> [dx, dy, dz]
        dx = c_neigh_vec[:, 0]
        dz = c_neigh_vec[:, 2]
        r_xz = np.sqrt(dx**2 + dz**2)
        
        # Filter: strictly enforce the cylindrical cutoff
        mask = r_xz <= self.d
        
        # Apply mask to reduce arrays to valid pairs only
        valid_central = central_indices[mask]
        valid_neighbor = neighbor_indices[mask]
        valid_r_xz = r_xz[mask]
        
        # Calculate weights: w = [1 - (r/d)^2]^2
        # Note: r_xz <= d ensures term inside bracket is >= 0
        w = (1.0 - (valid_r_xz / self.d)**2)**2
        
        # Get phi values for the neighbors
        phi_neighbors = phi[valid_neighbor]
        
        # Calculate terms to accumulate
        numerator_contrib = w * valid_r_xz * phi_neighbors
        denominator_contrib = w * valid_r_xz
        
        # Accumulate sums into arrays using np.add.at (fast, unbuffered summation)
        numerator_sum = np.zeros(data.particles.count)
        denominator_sum = np.zeros(data.particles.count)
        
        np.add.at(numerator_sum, valid_central, numerator_contrib)
        np.add.at(denominator_sum, valid_central, denominator_contrib)
        
        # Final division to get Phi (handle division by zero)
        Phi_values = np.zeros(data.particles.count)
        non_zero_mask = denominator_sum > 1e-9
        Phi_values[non_zero_mask] = numerator_sum[non_zero_mask] / denominator_sum[non_zero_mask]

        # Store the result
        data.particles_.create_property(self.final_property_name, data=Phi_values)

    def apply_modifier(self, node):
        node.modifiers.append(PythonScriptModifier(function=self._calculate_lop))

    def get_property_name(self):
        return self.final_property_name
        
# --- Helper for Miller Indices ---

def get_orientation_matrix(miller_x, miller_y, miller_z):
    """
    Converts three Miller indices (as lists/tuples) into a 3x3 orientation matrix.
    """
    vx = np.array(miller_x)
    vy = np.array(miller_y)
    vz = np.array(miller_z)
    
    # Normalize
    ux = vx / np.linalg.norm(vx)
    uy = vy / np.linalg.norm(vy)
    uz = vz / np.linalg.norm(vz)
    
    # Verify orthogonality
    if not np.allclose([np.dot(ux, uy), np.dot(ux, uz), np.dot(uy, uz)], 0, atol=1e-5):
        raise ValueError("Miller indices provided are not orthogonal.")
        
    return np.array([ux, uy, uz])

# --- New Structural Matching Modifier ---

class OrientationPhiModifier(PhaseModifierBase):
    """
    Phase identification using structural matching with known crystal orientation.
    Calculates Phi = sum(|r_i - r_fcc|^2) for the 12 nearest neighbors.
    """
    def __init__(self, lattice_constant=4.05, 
                 miller_x=[1, 0, 0], miller_y=[0, 1, 0], miller_z=[0, 0, 1], 
                 **kwargs):
        super().__init__(**kwargs)
        self.a = lattice_constant
        self.rotation_matrix = get_orientation_matrix(miller_x, miller_y, miller_z)
        self.final_property_name = 'Orientation_Phi'

    def _compute_phi_orient(self, frame, data):
        # 1. Define ideal FCC reference vectors (1/2 <110> family)
        ref_vecs = np.array([
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
        ]) * (self.a / 2.0)

        # 2. Rotate reference vectors to match simulation box orientation
        # We rotate the reference vectors from crystal-space to box-space
        ref_vecs = ref_vecs @ self.rotation_matrix.T

        # 3. Find 12 nearest neighbors
        finder = NearestNeighborFinder(N=12, data_collection=data)
        neigh_idx, neigh_vecs = finder.find_all() # neigh_vecs: (N_particles, 12, 3)

        # 4. Calculate Deviation Phi
        # We need to find the best match for each neighbor among the 12 reference vectors
        # dists_sq shape: (N_particles, 12_actual, 12_ideal)
        diff = neigh_vecs[:, :, np.newaxis, :] - ref_vecs[np.newaxis, np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=3)
        
        # For each actual neighbor, find the minimum distance to any ideal vector
        min_dists_sq = np.min(dists_sq, axis=2) 
        
        # Final Phi is the sum of these minimum distances for all 12 neighbors
        phi_values = np.sum(min_dists_sq, axis=1)

        # 5. Store as property
        data.particles_.create_property(self.final_property_name, data=phi_values)

    def apply_modifier(self, node):
        node.modifiers.append(PythonScriptModifier(function=self._compute_phi_orient))

    def get_property_name(self):
        return self.final_property_name
