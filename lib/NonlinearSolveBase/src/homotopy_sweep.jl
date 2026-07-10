"""
    HomotopySweep(; inner = nothing, nsteps = nothing, adaptive = true,
        initial_step_factor = 0.1, min_dλ = nothing, max_step_factor = 1.0,
        expand_factor = 2.0, expand_threshold = 2, expand_quality = 0.25,
        predictor = :secant)

Natural-parameter continuation solver for a [`SciMLBase.HomotopyProblem`](@ref). The
scalar continuation parameter ``λ`` is swept across the problem's `λspan`. The sweep
first solves the system at `λspan[1]` (for the canonical `(0, 1)` span, the
`simplified` system — the form the homotopy is designed to make solvable from a cold
start) starting from `u0`; each subsequent step fixes ``λ``, predicts a warm start by
extrapolating along the solution path, and corrects it by solving the resulting
standard nonlinear system with `inner`.

The inner solver is initialized once and re-driven each step through the
`init`/`reinit!`/`solve!` cache interface, so the continuation loop reuses the inner
solver's workspace (Jacobian buffers, linear-solver storage) instead of reconstructing
the solver every step; the sweep's own per-step state lives in a fixed set of
preallocated buffers.

The step size is governed by the classic success/failure heuristic of
predictor-corrector path tracking (see e.g. Timme, *Mixed precision path tracking for
polynomial homotopy continuation*, Advances in Computational Mathematics 47, 2021): a
failed corrector halves the λ increment and retries from the last accepted point, while
`expand_threshold` consecutive accepted steps grow the increment by `expand_factor`,
capped at `max_step_factor` of the span width. Expansion is additionally gated on the
quality of the secant prediction (a Deuflhard-style local error estimate): the step
only grows when the corrector's correction was small relative to how far the solution
moved, so the increment does not balloon right before a sharp turn in the path — where
an oversized step would be rejected only after the inner solver exhausts its iterations.
This lets the sweep crawl through ill-conditioned regions of the path and accelerate
back out of them while keeping trial-and-error rejections cheap.

Keyword arguments:

  - `inner`: the inner nonlinear algorithm; `nothing` selects NonlinearSolve's default
    polyalgorithm (NOT a hardcoded Newton).
  - `nsteps`: when given, the initial λ increment is the span width divided by `nsteps`
    instead of `initial_step_factor`. Required when `adaptive = false` (the steps are
    then fixed-size).
  - `adaptive`: when `true` (default), a step whose inner solve fails to converge halves
    the λ increment and retries, down to a floor of `min_dλ`, and consecutive successes
    expand the increment as described above.
  - `initial_step_factor`: the initial λ increment as a fraction of the `λspan` width;
    used when `nsteps` is not given.
  - `min_dλ`: the smallest λ increment bisection may reach; `nothing` (default) resolves
    to `sqrt(eps(typeof(λ)))` at solve time, so the floor scales with precision.
  - `max_step_factor`: the largest λ increment, as a fraction of the `λspan` width, that
    success expansion may reach. Must be in `(0, 1]`. Smaller values bound how far any
    single step can move along the path, which reduces the risk of the corrector
    converging to a different solution branch ("path jumping") on multi-branch problems.
  - `expand_factor`: the multiplier applied to the λ increment after `expand_threshold`
    consecutive successful steps. Must be ≥ 1; `1` disables expansion.
  - `expand_threshold`: the number of consecutive successful steps required before the
    increment is expanded. Must be ≥ 1. Larger values make regrowth more cautious after
    a bisection, avoiding repeated fail-shrink-regrow churn inside a hard region.
  - `expand_quality`: expansion additionally requires the secant prediction's error
    `‖u - u_predicted‖` to be at most `expand_quality` times the scale of the recent
    solution movement. The error is measured against the prediction the secant *would*
    have made regardless of the `predictor` setting, so the gate is active for both
    predictors. A step whose corrector reports convergence within 2 iterations passes
    the gate outright — the warm start was deep inside the convergence basin, which is
    the strongest evidence that a larger step is affordable (and the relative error
    measure is uninformative on stretches where the path barely moves). Must be
    positive; `Inf` disables the gate, leaving the unconditional success/failure
    heuristic.
  - `predictor`: `:secant` (default) extrapolates the initial guess for the next step
    linearly through the last two accepted points, so the corrector starts on the path
    tangent rather than at the previous solution; `:constant` warm-starts from the
    previous solution unchanged. The secant is trust-monitored: whenever its measured
    prediction error is no better than half that of the trivial constant prediction
    (as happens right after a sharp turn in the path, where the stale tangent points
    away from the path), or a step is rejected outright, subsequent steps fall back to
    the constant warm start until two consecutive accepted steps measure good secant
    quality again.

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
    max_step_factor
    expand_factor
    expand_threshold::Int
    expand_quality
    predictor::Symbol
end

function HomotopySweep(;
        inner = nothing, nsteps = nothing, adaptive = true,
        initial_step_factor = 0.1, min_dλ = nothing, max_step_factor = 1.0,
        expand_factor = 2.0, expand_threshold = 2, expand_quality = 0.25,
        predictor = :secant
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
    if !(0 < max_step_factor <= 1)
        throw(
            ArgumentError(
                "HomotopySweep `max_step_factor` must be in (0, 1], got $max_step_factor"
            )
        )
    end
    if expand_factor < 1
        throw(
            ArgumentError(
                "HomotopySweep `expand_factor` must be ≥ 1 (1 disables expansion), " *
                    "got $expand_factor"
            )
        )
    end
    if expand_threshold < 1
        throw(
            ArgumentError(
                "HomotopySweep `expand_threshold` must be ≥ 1, got $expand_threshold"
            )
        )
    end
    if !(expand_quality > 0)
        throw(
            ArgumentError(
                "HomotopySweep `expand_quality` must be positive (Inf disables the " *
                    "gate), got $expand_quality"
            )
        )
    end
    if predictor !== :secant && predictor !== :constant
        throw(
            ArgumentError(
                "HomotopySweep `predictor` must be :secant or :constant, got :$predictor"
            )
        )
    end
    return HomotopySweep(
        inner, nsteps, adaptive, initial_step_factor, min_dλ,
        max_step_factor, expand_factor, expand_threshold, expand_quality, predictor
    )
end

# Fixes λ as the trailing argument, exposing the standard nonlinear calling convention
# `(u, p)` / `(du, u, p)` to the inner solver. `λ` is mutable so the SAME function (and
# the SAME inner-solver cache built around it) is reused across every continuation
# step — advancing λ is a field write, not a new function/problem/solver allocation.
mutable struct FixLambda{F, T}
    const f::F
    λ::T
end
(fl::FixLambda)(args...) = fl.f(args..., fl.λ)

# Secant/constant predictor `u + s*(u - u_prev)`, written into the reused `dst` buffer
# for mutable arrays (no allocation) or returned fresh for immutable/StaticArray/scalar
# `u` (stack, no heap). `Utils.can_setindex` is a compile-time trait, so the branch is
# elided.
function _sweep_extrapolate!(dst, u, u_prev, s)
    if Utils.can_setindex(u)
        @. dst = u + s * (u - u_prev)
        return dst
    else
        return @. u + s * (u - u_prev)
    end
end

# Constant warm start: a copy of `u` into the reused `dst` buffer (mutable) or `u`
# itself (immutable — the inner solver cannot mutate it in place, so aliasing is
# harmless).
_sweep_warmstart!(dst, u) = Utils.can_setindex(u) ? (copyto!(dst, u); dst) : u

# Accept `sol_u` as the new current iterate, shifting the old current into `u_prev`. For
# mutable arrays this swaps the two buffers and copies in place (no allocation); for
# immutable `u` it reassigns (stack). Returns the new `(u, u_prev)`.
function _sweep_accept!(u, u_prev, sol_u)
    if Utils.can_setindex(u)
        u, u_prev = u_prev, u
        copyto!(u, sol_u)
        return u, u_prev
    else
        return sol_u, u
    end
end

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
    max_dλ = λT(alg.max_step_factor) * span     # carries span's sign, like dλ
    expand_factor = λT(alg.expand_factor)
    abs(dλ) > abs(max_dλ) && (dλ = max_dλ)
    u = copy(prob.u0)

    # One inner-solver cache, built once around the mutable `FixLambda` and reused for
    # every solve of the sweep (the anchor below and each continuation step): advancing
    # λ is a field write plus a `reinit!` with the new warm start, so the inner solver's
    # workspace is reused instead of reconstructed. `guess` is handed to the cache and
    # may be iterated in place when aliasing is forwarded; `u` is a separately-owned
    # buffer the inner solver never writes, so a FAILED attempt leaves the last
    # converged iterate intact for retries and the failure returns below.
    fixλ = FixLambda(prob.f, λ)
    fλ = SciMLBase.NonlinearFunction{iip}(fixλ)
    guess = copy(u)
    cache = init(
        NonlinearProblem{iip}(fλ, guess, prob.p), alg.inner, args...;
        prob.kwargs..., kwargs...
    )

    # Anchor: solve the system at λ = λspan[1] from u0 BEFORE stepping. For the
    # canonical (0, 1) span this is the pure `simplified` system — the one the
    # homotopy contract is designed to make solvable from a cold start (OMC's
    # reference implementation also solves λ = 0 first). Without this anchor the
    # first inner solve runs at λ0 + dλ warm-started from u0, so a poor u0 can
    # converge onto the wrong branch and the sweep then tracks that branch all
    # the way to a wrong root with a success retcode.
    last_sol = CommonSolve.solve!(cache)
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

    # Reused across every step: `u`/`u_prev` are the last two accepted iterates (their
    # buffers are swapped, never reallocated); `virtual` is scratch for the secant
    # quality gate. λ_prev == λ means there is no history yet and the predictor falls
    # back to a constant warm start.
    u_prev = copy(u)
    virtual = Utils.safe_similar(u)
    λ_prev = λ
    streak = 0
    # Consecutive accepted steps whose measured secant quality was good. The secant is
    # only used while trust ≥ 2: requiring sustained evidence (hysteresis) keeps one
    # coincidentally good prediction inside a curved region from re-arming a stale
    # tangent. Initialized at 2 so the secant engages as soon as history exists.
    trust = 2
    disp_prev = zero(λT)

    while true
        next_λ = abs(λend - λ) <= abs(dλ) ? λend : λ + dλ
        if next_λ == λ && next_λ != λend
            # dλ underflowed below eps(λ) mid-continuation: no further progress is
            # possible (the zero-width span is already handled by the anchor above).
            return SciMLBase.build_solution(
                prob, alg, u, nothing; retcode = ReturnCode.Stalled
            )
        end
        used_secant = alg.predictor === :secant && trust >= 2 && λ_prev != λ
        guess = if used_secant
            # Bisection shrinks `next_λ - λ` and with it the extrapolation length, so a
            # prediction that overshoots degrades gracefully toward the constant guess.
            s = (next_λ - λ) / (λ - λ_prev)
            _sweep_extrapolate!(guess, u, u_prev, s)
        else
            _sweep_warmstart!(guess, u)
        end
        fixλ.λ = next_λ
        SciMLBase.reinit!(cache, guess)
        last_sol = CommonSolve.solve!(cache)

        if SciMLBase.successful_retcode(last_sol)
            # The secant prediction error θ (relative to the recent solution movement)
            # is a cheap local error estimate: it grows with the path's curvature times
            # dλ², so a large θ means the path is turning and a stale tangent would
            # land the next prediction far off the path, failing expensively. It is
            # measured against the prediction the secant WOULD have made even on steps
            # that warm-started constantly, so trust can be regained once two accepted
            # points lie past a sharp turn. The scale includes the previous step's
            # displacement and an absolute floor so that a flat stretch of the path
            # (where the displacement is rounding noise) doesn't read as distrust.
            θ = nothing
            if λ_prev != λ
                # recomputed from scratch (never reuse `guess`): the inner solver may
                # have iterated in place in the guess buffer when aliasing is forwarded
                sv = (next_λ - λ) / (λ - λ_prev)
                virtual = _sweep_extrapolate!(virtual, u, u_prev, sv)
                correction = Utils.norm_op(L2_NORM, -, last_sol.u, virtual)
                disp = Utils.norm_op(L2_NORM, -, last_sol.u, u)
                scale = max(disp, disp_prev, sqrt(eps(λT)) * (1 + L2_NORM(last_sol.u)))
                θ = correction / scale
                # the secant only earns its keep when it predicts at least twice as
                # well as the trivial constant prediction (whose θ is exactly 1)
                trust = θ < 1 / 2 ? trust + 1 : 0
                disp_prev = disp
            else
                disp_prev = Utils.norm_op(L2_NORM, -, last_sol.u, u)
            end
            # accept: swap `u`↔`u_prev` and copy the solution into `u` (no allocation).
            u, u_prev = _sweep_accept!(u, u_prev, last_sol.u)
            λ_prev = λ
            λ = next_λ
            λ == λend && break
            if alg.adaptive
                streak += 1
                # Growth requires a streak of successes (the classic heuristic) plus
                # evidence the corrector has headroom: either a small relative
                # prediction error (the quality gate) or a corrector that converged
                # almost immediately. The iteration count covers paths that flatten
                # exponentially, where θ stays at a constant mediocre value while the
                # absolute corrections — and hence the corrector work — become
                # negligible. The streak is NOT reset on a vetoed expansion, so growth
                # resumes on the first step whose evidence recovers.
                corrector_cheap = last_sol.stats !== nothing &&
                    last_sol.stats.nsteps <= 2
                if streak >= alg.expand_threshold &&
                        (θ === nothing || θ <= λT(alg.expand_quality) || corrector_cheap)
                    grown = expand_factor * dλ
                    dλ = abs(grown) > abs(max_dλ) ? max_dλ : grown
                    streak = 0
                end
            end
        elseif alg.adaptive && abs(dλ) / 2 >= min_dλ
            dλ = dλ / 2          # bisect; retry from the same λ (do not advance)
            streak = 0
            # a rejected step is evidence against the tangent: bisection retries (and
            # the steps right after) warm-start constantly until quality re-accumulates
            trust = 0
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

# A HomotopyProblem with no algorithm defaults to the staged polyalgorithm (sweep first,
# pseudo-arclength fallback). This lets the generic `solve(initprob, nothing)` path
# (e.g. SciMLBase OverrideInit with a default nlsolve) route a homotopy initialization
# problem to the most robust default automatically.
function CommonSolve.solve(prob::SciMLBase.HomotopyProblem, ::Nothing, args...; kwargs...)
    return solve(prob, HomotopyPolyAlgorithm(), args...; kwargs...)
end

# The zero-argument form `solve(prob)` must route the same way instead of falling into
# the generic concrete-problem machinery (which has no HomotopyProblem path).
function CommonSolve.solve(prob::SciMLBase.HomotopyProblem; kwargs...)
    return solve(prob, HomotopyPolyAlgorithm(); kwargs...)
end
