"""
    HomotopySweep(; inner = nothing, nsteps = nothing, adaptive = true,
        initial_step_factor = 0.1, min_dλ = nothing)

Natural-parameter continuation solver for a [`SciMLBase.HomotopyProblem`](@ref). The
scalar continuation parameter ``λ`` is swept across the problem's `λspan`. The sweep
first solves the system at `λspan[1]` (for the canonical `(0, 1)` span, the
`simplified` system — the form the homotopy is designed to make solvable from a cold
start) starting from `u0`; each subsequent step fixes ``λ``, solves the resulting
standard nonlinear system with `inner`, and warm-starts from the previous step's
solution.

Keyword arguments:

  - `inner`: the inner nonlinear algorithm; `nothing` selects NonlinearSolve's default
    polyalgorithm (NOT a hardcoded Newton).
  - `nsteps`: when given, the initial λ increment is the span width divided by `nsteps`
    instead of `initial_step_factor`. Required when `adaptive = false` (the steps are
    then fixed-size).
  - `adaptive`: when `true` (default), a step whose inner solve fails to converge halves
    the λ increment and retries, down to a floor of `min_dλ`.
  - `initial_step_factor`: the initial λ increment as a fraction of the `λspan` width;
    used when `nsteps` is not given.
  - `min_dλ`: the smallest λ increment bisection may reach; `nothing` (default) resolves
    to `sqrt(eps(typeof(λ)))` at solve time, so the floor scales with precision.

When the sweep cannot reach the end of `λspan`, the returned solution carries a failure
retcode: its `u` is the last converged iterate (at some ``λ`` short of `λspan[2]`, or
`u0` itself if the initial `λspan[1]` anchor solve failed), while `resid` comes from the
failed step (`nothing` on the `ReturnCode.Stalled` path, where no step ran).

This is the embedding-homotopy / continuation analogue used to robustly initialize
systems whose target form is hard to solve cold; it is unrelated to the polynomial
`HomotopyContinuationJL`.
"""
@concrete struct HomotopySweep <: AbstractNonlinearSolveAlgorithm
    inner
    nsteps
    adaptive::Bool
    initial_step_factor
    min_dλ
end

function HomotopySweep(;
        inner = nothing, nsteps = nothing, adaptive = true,
        initial_step_factor = 0.1, min_dλ = nothing
    )
    if nsteps !== nothing && nsteps < 1
        throw(ArgumentError("HomotopySweep `nsteps` must be ≥ 1, got $nsteps"))
    end
    if !adaptive && nsteps === nothing
        throw(
            ArgumentError(
                "HomotopySweep with `adaptive = false` takes fixed-size λ steps, so an " *
                    "explicit `nsteps` is required."
            )
        )
    end
    if !(0 < initial_step_factor <= 1)
        throw(
            ArgumentError(
                "HomotopySweep `initial_step_factor` must be in (0, 1], got $initial_step_factor"
            )
        )
    end
    if min_dλ !== nothing && min_dλ <= 0
        # min_dλ = 0 would make the bisection guard always true → dλ halves forever
        throw(ArgumentError("HomotopySweep `min_dλ` must be positive, got $min_dλ"))
    end
    return HomotopySweep(inner, nsteps, adaptive, initial_step_factor, min_dλ)
end

# Fixes λ as the trailing argument, exposing the standard nonlinear calling convention
# `(u, p)` / `(du, u, p)` to the inner solver. A named struct rather than a per-step
# closure, so every continuation step has the same function type and the inner solver's
# compilation is reused across steps.
struct FixLambda{F, T}
    f::F
    λ::T
end
(fl::FixLambda)(args...) = fl.f(args..., fl.λ)

function CommonSolve.solve(
        prob::SciMLBase.HomotopyProblem{uType, iip},
        alg::HomotopySweep, args...; kwargs...
    ) where {uType, iip}
    λ0, λ1 = prob.λspan
    λ = float(λ0)
    λT = typeof(λ)
    λend = λT(λ1)
    span = λend - λ
    dλ = alg.nsteps === nothing ? λT(alg.initial_step_factor) * span : span / alg.nsteps
    min_dλ = alg.min_dλ === nothing ? sqrt(eps(λT)) : λT(alg.min_dλ)
    u = copy(prob.u0)

    # Anchor: solve the system at λ = λspan[1] from u0 BEFORE stepping. For the
    # canonical (0, 1) span this is the pure `simplified` system — the one the
    # homotopy contract is designed to make solvable from a cold start (OMC's
    # reference implementation also solves λ = 0 first). Without this anchor the
    # first inner solve runs at λ0 + dλ warm-started from u0, so a poor u0 can
    # converge onto the wrong branch and the sweep then tracks that branch all
    # the way to a wrong root with a success retcode.
    anchor_f = SciMLBase.NonlinearFunction{iip}(FixLambda(prob.f, λ))
    last_sol = solve(
        NonlinearProblem{iip}(anchor_f, copy(u), prob.p),
        alg.inner, args...; prob.kwargs..., kwargs...
    )
    if !SciMLBase.successful_retcode(last_sol)
        # the λ = λspan[1] system itself failed from u0: the homotopy premise is
        # broken, so no continuation is possible. `u` stays u0.
        return SciMLBase.build_solution(
            prob, alg, u, last_sol.resid;
            retcode = last_sol.retcode, original = last_sol
        )
    end
    u = copy(last_sol.u)
    # Zero-width λspan (λ0 == λend): the anchor IS the single target solve.
    λ == λend && return SciMLBase.build_solution(
        prob, alg, u, last_sol.resid; retcode = ReturnCode.Success
    )

    while true
        next_λ = abs(λend - λ) <= abs(dλ) ? λend : λ + dλ
        if next_λ == λ && next_λ != λend
            # dλ underflowed below eps(λ) mid-continuation: no further progress is
            # possible (the zero-width span is already handled by the anchor above).
            return SciMLBase.build_solution(
                prob, alg, u, nothing; retcode = ReturnCode.Stalled
            )
        end
        fλ = SciMLBase.NonlinearFunction{iip}(FixLambda(prob.f, next_λ))
        # The inner solver may iterate directly in its u0 buffer (e.g. when
        # `alias = NonlinearAliasSpecifier(alias_u0 = true)` is forwarded), and a FAILED
        # attempt leaves diverged garbage there; hand over a copy so `u` always remains
        # the last converged iterate for retries and for the failure return below.
        inner_prob = NonlinearProblem{iip}(fλ, copy(u), prob.p)
        last_sol = solve(inner_prob, alg.inner, args...; prob.kwargs..., kwargs...)

        if SciMLBase.successful_retcode(last_sol)
            # defensive: some inner solvers may return views into persistent workspace;
            # keep our iterate independently owned.
            u = copy(last_sol.u)
            λ = next_λ
            λ == λend && break
        elseif alg.adaptive && abs(dλ) / 2 >= min_dλ
            # conservative: step size is not restored after a bisection
            dλ = dλ / 2          # bisect; retry from the same λ (do not advance)
        else
            # on failure: u is the last converged iterate (λ<λ1); resid is from the failed step (advisory)
            return SciMLBase.build_solution(
                prob, alg, u, last_sol.resid;
                retcode = last_sol.retcode, original = last_sol
            )
        end
    end

    return SciMLBase.build_solution(
        prob, alg, u, last_sol.resid; retcode = ReturnCode.Success
    )
end

# A HomotopyProblem with no algorithm defaults to the continuation sweep. This lets the
# generic `solve(initprob, nothing)` path (e.g. SciMLBase OverrideInit with a default
# nlsolve) route a homotopy initialization problem to HomotopySweep automatically.
function CommonSolve.solve(prob::SciMLBase.HomotopyProblem, ::Nothing, args...; kwargs...)
    return solve(prob, HomotopySweep(), args...; kwargs...)
end

# The zero-argument form `solve(prob)` must route the same way instead of falling into
# the generic concrete-problem machinery (which has no HomotopyProblem path).
function CommonSolve.solve(prob::SciMLBase.HomotopyProblem; kwargs...)
    return solve(prob, HomotopySweep(); kwargs...)
end
