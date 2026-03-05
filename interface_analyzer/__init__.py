"""
Interface Analyzer package for processing molecular dynamics configurations
and performing Capillary Fluctuation Method (CFM) analysis.
"""

from .analysis import (
    analyze_by_CSP,
    analyze_by_PTM,
    analyze_by_LOP,
    analyze_by_OrientationPhi,  # New: Orientation-based wrapper
    analyze_cfm,
    plot_cfm_k2_single,
    analyze_cfm_fit_sensitivity,
    analyze_by_custom_modifier,
    LOP_analysis,
    Orientation_analysis       # New: Grid-based orientation analysis
)

from .modifiers import (
    PhaseModifierBase, 
    CSPModifier, 
    PTMModifier, 
    LOPModifier, 
    OrientationPhiModifier,    # New: Modifier class
    get_orientation_matrix     # New: Matrix helper
)

__all__ = [
    'analyze_by_CSP',
    'analyze_by_PTM',
    'analyze_by_LOP',
    'analyze_by_OrientationPhi',
    'analyze_cfm',
    'plot_cfm_k2_single',
    'analyze_cfm_fit_sensitivity',
    'analyze_by_custom_modifier',
    'LOP_analysis',
    'Orientation_analysis',
    'PhaseModifierBase',
    'CSPModifier',
    'PTMModifier',
    'LOPModifier',
    'OrientationPhiModifier',
    'get_orientation_matrix'
]

# Helper functions for binning and interface detection are kept private.
