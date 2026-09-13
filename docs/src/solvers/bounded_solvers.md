# [Bounded Solvers](@id bounded-solvers)

Use `solve(prob)` for a problem with `lb` or `ub`. The bounded default works in the
original coordinates: [`FastShortcutBoundedPolyalg`](@ref) tries `BoundedTrustRegion()`
and then `BoundedGaussNewton()` with projected backtracking if needed. The
[bound-constraints tutorial](../tutorials/bound_constraints.md) shows how to construct root and least-squares problems.

## Choosing a solver

Start with the default when the problem's numerical behavior is unknown. For repeated
solves of the same model, benchmark individual methods after checking residuals and
stationarity: in the bounded solver benchmark in
[SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl),
`BoundedTrustRegion()` has low overhead, while projected `BoundedGaussNewton()` also
handles its difficult underdetermined case. For operator-based problems, also benchmark
projected `BoundedGaussNewton()` explicitly: it was faster on the diffusion operator
cases in that benchmark. Reuse a cache with `init`/`reinit!`/`solve!`;
`retain_best = true` can reuse a successful fallback stage across parameter sweeps, with
periodic retries of the earlier stages.

When a `postcondition` corrector is supplied, the default restricts the sequence to
`BoundedTrustRegion`, the stage that supports iterate correction.

The native methods share the Jacobian, linear-solver, and bound-handling machinery.
Their step models and globalization strategies differ:

| Method | Step and globalization |
|:--|:--|
| [`BoundedTrustRegion`](@ref) | Projected dogleg step with a spherical trust region |
| [`TrustRegionReflective`](@ref) | Coleman–Li distance-to-bound scaling and reflected trial steps |
| [`BoundedLevenbergMarquardt`](@ref) | Box-constrained damped least-squares model with a feasible line search |
| [`Dogbox`](@ref) | Dogleg step inside a rectangular trust region intersected with the bounds |
| [`BoundedGaussNewton`](@ref) | Active-set Gauss–Newton with projected line search, trust region, or their combination |

## Feasibility and convergence

Supply a real floating-point state. A feasible initial guess is recommended; the
automatic default and explicit polyalgorithms containing only native bounded stages
project an out-of-bounds guess onto the box. Individual native algorithms require a
feasible initial guess. This projection also applies to auxiliary initialization problems,
after their initialization callbacks update the guess. Exact-bound initial
values and fixed coordinates are supported. `TrustRegionReflective` moves nonfixed
exact-bound initial values into the strict interior; the other native methods permit
exact-bound iterates. Bounds are local constraints, not a global-search strategy.

For a `NonlinearProblem`, success still requires a small residual. For a
`NonlinearLeastSquaresProblem`, projected-gradient stationarity can establish success
even when the residual is nonzero. A stationary constrained least-squares point is not
necessarily a root or a global minimum.

```@example bounded_solvers
using NonlinearSolve, SciMLBase
prob = NonlinearLeastSquaresProblem(
    (u, p) -> [u[1] - 2.0, 1.0], [0.0]; lb = -1.0, ub = 1.0
)
sol = solve(prob)
@assert SciMLBase.successful_retcode(sol)
@assert sol.u ≈ [1.0]
sol.resid
```

## Sparse and operator Jacobians

A sparse `jac_prototype` remains sparse. A Krylov linear solver uses Jacobian-vector
and transpose-Jacobian-vector operators unless `concrete_jac = true` is requested.
The native methods reuse the selected linear solver and support `jvp_autodiff`,
`vjp_autodiff`, analytic `jvp`/`vjp` callbacks, and preconditioning.

```@example bounded_solvers
using LinearSolve, SparseArrays
f! = (r, u, p) -> (r .= u .- p)
nf = NonlinearFunction(f!; jac_prototype = spdiagm(0 => ones(20)))
prob_sparse = NonlinearLeastSquaresProblem(
    nf, fill(0.5, 20), collect(range(-0.5, 1.5; length = 20)); lb = 0.0, ub = 1.0
)
sol_sparse = solve(prob_sparse, BoundedGaussNewton())
sol_operator = solve(prob_sparse, BoundedGaussNewton(linsolve = KrylovJL_LSMR()))
@assert SciMLBase.successful_retcode(sol_sparse)
@assert SciMLBase.successful_retcode(sol_operator)
@assert maximum(abs, sol_sparse.u - sol_operator.u) < 1e-6
```

Choose a linear solver that supports rank deficiency or underdetermined systems when
needed. `KrylovJL_LSMR()` solves the rectangular least-squares models; linear solvers
requiring square systems use normal equations. A successful minimum-norm linear solve
does not by itself ensure fast nonlinear convergence.

For a scalar state with an array residual, or an array state with a scalar residual,
select `BoundedGaussNewton()` explicitly. The generalized dogleg stage used by the
default currently requires matching scalar/array categories.

Analytic Jacobians and AD backends follow the usual solver interface. Native finite
differences stay inside finite bounds and do not perturb fixed coordinates.

## Explicit algorithms without native bound support

An explicitly selected algorithm such as `NewtonRaphson()` or `LevenbergMarquardt()`
uses an automatic variable transformation. Two-sided bounds use a logistic map;
one-sided bounds use an exponential map. This is useful when a particular unbounded
algorithm is required, but the transformed derivative can become small near an active
bound. Native methods operate directly on the box and can reach its boundary.

## Solver API

```@docs
FastShortcutBoundedPolyalg
BoundedTrustRegion
TrustRegionReflective
BoundedLevenbergMarquardt
Dogbox
BoundedGaussNewton
```
