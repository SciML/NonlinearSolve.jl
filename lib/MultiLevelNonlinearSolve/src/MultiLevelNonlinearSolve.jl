"""
    MultiLevelNonlinearSolve

Multi-level Newton solvers for nonlinear systems whose unknowns split into a globally coupled
block `ū` and an internal block `q` that decouples into one small independent problem per
point.

[`MultiLevelNewton`](@ref) eliminates `q` at fixed `ū` and takes a Newton step on the
Schur-condensed system `S·δū = -R̄`, with `S` assembled from per-point correctors. See the
Multi-Level Newton tutorial for the trial/commit contract, the four accuracy knobs, and a
worked example.

```julia
f = MultiLevelNonlinearFunction(
    NonlinearFunction(Rbar!; jac = assemble_S!, jac_prototype = S);
    primary = 1:n̄, internal = (n̄ + 1):n, commit_internal!
)
sol = solve(NonlinearProblem(f, u0, p), MultiLevelNewton())   # sol.u is the full [ū; q]
```
"""
module MultiLevelNonlinearSolve

using ConcreteStructs: @concrete
using Reexport: @reexport

using LinearAlgebra: LinearAlgebra, rmul!
using LinearSolve: LinearSolve
using NonlinearSolveBase: NonlinearSolveBase, AbstractNonlinearSolveAlgorithm,
    AbstractNonlinearSolveCache, InternalAPI, NonlinearVerbosity, Utils,
    get_timer_output, @static_timeit
using NonlinearSolveFirstOrder: NonlinearSolveFirstOrder, NewtonRaphson
using SciMLBase: SciMLBase, AbstractNonlinearProblem, NLStats, ReturnCode
using SciMLOperators: SciMLOperators

include("local_tolerance.jl")
include("ensemble.jl")
include("function.jl")
include("solve.jl")
include("variant_a.jl")

# `NonlinearSolveFirstOrder` comes along because the `global_solver` is part of this
# package's own API surface — `NewtonRaphson()` is the default and `TrustRegion()` the usual
# alternative, so `using MultiLevelNonlinearSolve` has to be enough to name them.
@reexport using SciMLBase, NonlinearSolveBase, NonlinearSolveFirstOrder

export MultiLevelNewton, MultiLevelNonlinearFunction, LocalToleranceSchedule,
    LocalForcingParameters, local_tolerance, user_parameters, ncommits,
    LocalEnsemble, ensemble_foreach, LocalStateBuffer, committed_state, trial_state,
    commit_local_state!,
    fullspace_problem, SchurOperator, CondensedFactorization, MultiLevelProjection

end
