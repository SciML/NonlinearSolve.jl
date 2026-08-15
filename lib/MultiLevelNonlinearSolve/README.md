# MultiLevelNonlinearSolve.jl

Multi-level Newton solvers for nonlinear systems whose unknowns split into a global block
`ū` and an internal block `q` that decouples into one small independent problem per point —
the shape of FEM problems with internal variables (plasticity, damage, viscoelasticity).

`MultiLevelNewton` eliminates `q` at fixed `ū` with a local solve per point, then takes a
Newton step on the Schur-condensed system `S·δū = -R̄`. The cross blocks are never formed:
the user assembles `S` from per-point correctors, exactly as an element tangent is assembled.

See the [documentation](https://docs.sciml.ai/NonlinearSolve/stable/) for details.
