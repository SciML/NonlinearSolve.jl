using NonlinearSolve

using SciMLBase

import NonlinearSolveBase

# Root-cause type-stability regression for the continuation drivers.
#
# The drivers assemble their returned solution from an inner solve whose return type
# inference gives up: `_sweep_exempt_solve` (and its arclength twin) builds a
# `NonlinearProblem` whose residual is a `FixLambda` wrapper, and the `AutoSpecialize`
# `FunctionWrappersWrapper` prototype inference bails on that wrapper-of-a-wrapper, so
# `solve(inner_prob, inner)` widens to `Any`. Storing that `Any`-typed solution into the
# driver's own returned solution via `original = last_sol` pinned the returned solution's
# `original` type-slot to `Any`, making the whole driver `solve` infer as `Any` — every
# downstream `.u`/`.resid`/`.retcode` read on the returned solution then dynamically
# dispatched. The fix is twofold and mirrors `NonlinearSolvePolyAlgorithm`:
#   * the inner `NonlinearFunction` is built `FullSpecialize`, so the inner solve infers
#     (no `Any` from `_sweep_exempt_solve` / `_arclength_fixed_solve`); and
#   * `original` is gated behind a `store_original::Val` (default `Val(false)`), so the
#     returned solution's `original` slot is pinned to `Nothing` and the type is concrete
#     unless the user explicitly opts in.

H!(du, u, p, λ) = (du[1] = (1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1]); nothing)
prob = HomotopyProblem(H!, [4.0], [4.0])

# The inner exempt/fixed solves must no longer infer as `Any` (this is what poisoned the
# driver's returned type). `Nothing` is the concrete type of the default `alg.inner`.
@test code_typed(
    NonlinearSolveBase._sweep_exempt_solve,
    Tuple{typeof(prob), Nothing, Vector{Float64}, Float64}; optimize = true
)[1].second !== Any
@test code_typed(
    NonlinearSolveBase._arclength_fixed_solve,
    Tuple{typeof(prob), Nothing, Vector{Float64}, Float64}; optimize = true
)[1].second !== Any

# By default the drivers drop `original`, so their returned solution is concretely typed
# (its `original` slot is `Nothing`, not `Any`), and the full `solve` no longer infers as
# `Any`.
for alg in (HomotopySweep(), ArcLengthContinuation())
    sol = solve(prob, alg)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 2.0 atol = 1.0e-6
    @test isconcretetype(typeof(sol))
    @test fieldtype(typeof(sol), :original) === Nothing
    @test sol.original === nothing
    @test code_typed(
        SciMLBase.solve, Tuple{typeof(prob), typeof(alg)}; optimize = true
    )[1].second !== Any
end

# The `store_original` field is carried and defaults to `Val(false)`.
@test HomotopySweep().store_original === Val(false)
@test ArcLengthContinuation().store_original === Val(false)
@test HomotopyPolyAlgorithm().store_original === Val(false)
@test HomotopySweep(; store_original = Val(true)).store_original === Val(true)
@test ArcLengthContinuation(; store_original = Val(true)).store_original === Val(true)
@test HomotopyPolyAlgorithm(; store_original = Val(true)).store_original === Val(true)

# Opting in keeps `original` reachable on the failure paths that populate it. A homotopy
# whose target system has no real solution reachable from u0 fails the anchor solve, which
# is where the driver attaches `original`. With `store_original = Val(true)` the payload is
# retained; with the default it is dropped (slot pinned to `Nothing`).
Hfail!(du, u, p, λ) = (du[1] = (1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 + p[1]); nothing)
failprob = HomotopyProblem(Hfail!, [1.0], [1.0])
sfail_keep = solve(failprob, HomotopySweep(; store_original = Val(true)))
@test !SciMLBase.successful_retcode(sfail_keep)
@test sfail_keep.original !== nothing
@test sfail_keep.original isa SciMLBase.AbstractNonlinearSolution

sfail_drop = solve(failprob, HomotopySweep())
@test !SciMLBase.successful_retcode(sfail_drop)
@test sfail_drop.original === nothing
@test fieldtype(typeof(sfail_drop), :original) === Nothing
