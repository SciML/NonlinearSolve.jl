# MultiLevelNonlinearSolve.jl

Multi-level Newton solvers for nonlinear systems whose unknowns split into a globally coupled
block and per-point internal variables — the shape of FEM problems with internal variables
(plasticity, damage, viscoelasticity).

The internal variables are eliminated at fixed global state, one small solve per point, and
the global Newton step is taken on the Schur-condensed system. The cross blocks are never
formed: you assemble the condensed tangent from per-point correctors, the way an element
tangent is assembled.

See the [Multi-Level Newton tutorial](https://docs.sciml.ai/NonlinearSolve/stable/tutorials/multilevel_newton/)
for the contract the callbacks are bound by, the accuracy knobs, and a worked example.
