"""
    HomotopySweep(; inner = nothing, nsteps = nothing, adaptive = true,
        initial_step_factor = 0.1, min_dλ = nothing, max_step_factor = 1.0,
        expand_factor = 2.0, expand_threshold = 2, expand_quality = 0.25,
        predictor = :secant)

Natural-parameter continuation solver for a [`SciMLBase.HomotopyProblem`](@ref). The
scalar continuation parameter ``λ`` is swept across the problem's `λspan`; each step
fixes ``λ``, predicts an initial guess by extrapolating along the solution path, and
corrects it by solving the resulting standard nonlinear system with `inner`.

The step size is governed by the classic success/failure heuristic of
predictor-corrector path tracking (see e.g. Timme, *Mixed precision path tracking for
polynomial homotopy continuation*, Adv. Comput. Math. 47 (2021)): a failed corrector
halves the λ increment and retries from the last accepted point, while
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
    consecutive successful steps. Must be ≥ 1; `1` disables expansion entirely.
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
retcode: its `u` is the last converged iterate (at some ``λ`` short of `λspan[2]`),
while `resid` comes from the failed step (`nothing` on the `ReturnCode.Stalled` path,
where no step ran).

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
    max_dλ = λT(alg.max_step_factor) * span     # carries span's sign, like dλ
    expand_factor = λT(alg.expand_factor)
    abs(dλ) > abs(max_dλ) && (dλ = max_dλ)
    u = copy(prob.u0)
    # last two accepted points define the secant predictor; λ_prev == λ means there is
    # no history yet and the predictor falls back to a constant warm start.
    u_prev = u
    λ_prev = λ
    streak = 0
    # Consecutive accepted steps whose measured secant quality was good. The secant is
    # only used while trust ≥ 2: requiring sustained evidence (hysteresis) keeps one
    # coincidentally good prediction inside a curved region from re-arming a stale
    # tangent. Initialized at 2 so the secant engages as soon as history exists.
    trust = 2
    disp_prev = zero(λT)
    local last_sol

    while true
        next_λ = abs(λend - λ) <= abs(dλ) ? λend : λ + dλ
        if next_λ == λ && next_λ != λend
            # dλ underflowed below eps(λ): no further progress is possible. λend is
            # excluded so that a zero-width λspan still solves the target system once.
            return SciMLBase.build_solution(
                prob, alg, u, nothing; retcode = ReturnCode.Stalled
            )
        end
        fλ = SciMLBase.NonlinearFunction{iip}(FixLambda(prob.f, next_λ))
        # The inner solver may iterate directly in its u0 buffer (e.g. when
        # `alias = NonlinearAliasSpecifier(alias_u0 = true)` is forwarded), and a FAILED
        # attempt leaves diverged garbage there; hand over a freshly owned guess so `u`
        # always remains the last converged iterate for retries and for the failure
        # return below.
        used_secant = alg.predictor === :secant && trust >= 2 && λ_prev != λ
        guess = if used_secant
            # Bisection shrinks `next_λ - λ` and with it the extrapolation length, so a
            # prediction that overshoots degrades gracefully toward the constant guess.
            s = (next_λ - λ) / (λ - λ_prev)
            @. u + s * (u - u_prev)
        else
            copy(u)
        end
        inner_prob = NonlinearProblem{iip}(fλ, guess, prob.p)
        last_sol = solve(inner_prob, alg.inner, args...; prob.kwargs..., kwargs...)

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
                virtual = @. u + sv * (u - u_prev)
                correction = L2_NORM(last_sol.u .- virtual)
                disp = L2_NORM(last_sol.u .- u)
                scale = max(disp, disp_prev, sqrt(eps(λT)) * (1 + L2_NORM(last_sol.u)))
                θ = correction / scale
                # the secant only earns its keep when it predicts at least twice as
                # well as the trivial constant prediction (whose θ is exactly 1)
                trust = θ < 1 / 2 ? trust + 1 : 0
                disp_prev = disp
            else
                disp_prev = L2_NORM(last_sol.u .- u)
            end
            u_prev = u
            λ_prev = λ
            # defensive: some inner solvers may return views into persistent workspace;
            # keep our iterate independently owned.
            u = copy(last_sol.u)
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
