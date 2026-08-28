# Replica-resolved cubic-fit validation input

`cubic_stiffness_replicas.csv` is a small, deterministic **synthetic
validation fixture**.  It contains three whole-replica stiffness estimates for
each primary direction: `100_010`, `110_001`, and `110_1m10`.

The replica means were constructed from the known parameter vector
`(gamma0, gamma0*epsilon1, gamma0*epsilon2) = (103.0, 5.0, -0.28)` mJ m^-2.
The replica offsets have zero mean separately within each direction.  It is
therefore suitable for testing exact inversion and the propagation of
between-replica variation, but it is not a measurement from the bundled
20-frame dump trajectory and must not be reported as a physical result.

For a production analysis, replace this file with one row per completed MD
replica.  Each `stiffness_mJ_m2` must be produced with the same CFM estimator,
temperature definition, fit window, and post-processing parameters.
