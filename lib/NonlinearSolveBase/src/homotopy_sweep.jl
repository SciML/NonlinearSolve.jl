"""
    HomotopySweep(; inner = nothing, nsteps = nothing, adaptive = true,
        initial_step_factor = 0.1, min_dλ = nothing, max_step_factor = 1.0,
        expand_factor = 2.0, expand_threshold = 2, expand_quality = 0.25,
        predictor = :secant, tracking_maxiters = 10, maxsteps = 10000)

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
On success the growth multiplier is additionally scaled by the corrector's iteration
count (the AUTO-07p `ADPTDS` bands): a near-free corrector earns the full
`expand_factor`, moderate effort earns milder growth, effort past a quarter of the
iteration budget holds the increment, and a success that nearly exhausted the budget
proactively halves it — the corrector working that hard on an *accepted* step is the
earliest warning that the next full-size step will be rejected. This lets the sweep
crawl through ill-conditioned regions of the path and accelerate back out of them
while keeping trial-and-error rejections cheap.

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
  - `tracking_maxiters`: iteration cap for the inner solver on interior tracking
    steps (default 10, in the range used by MatCont,
    HomotopyContinuation.jl, and OpenModelica; `nothing` disables). A rejected step retries at half the
    increment from a warm start, so failing fast is far cheaper than exhausting the
    inner solver's full budget. Never applied to the `λspan[1]` anchor solve or the
    final step landing on `λspan[2]`; an explicit user-passed `maxiters` always wins.
  - `maxsteps`: a hard cap on the total number of predictor-corrector attempts
    (accepted steps plus bisection retries). Exceeding it returns a
    `ReturnCode.MaxIters` failure carrying the last converged iterate. Must be ≥ 1.

When the sweep cannot reach the end of `λspan`, the returned solution carries a failure
retcode: its `u` is the last converged iterate (at some ``λ`` short of `λspan[2]`, or
`u0` itself if the initial `λspan[1]` anchor solve failed), while `resid` comes from the
failed step (`nothing` on the `ReturnCode.Stalled` and `ReturnCode.MaxIters` paths,
where no failed step is available).

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
    tracking_maxiters
    maxsteps::Int
end

function HomotopySweep(;
        inner = nothing, nsteps = nothing, adaptive = true,
        initial_step_factor = 0.1, min_dλ = nothing, max_step_factor = 1.0,
        expand_factor = 2.0, expand_threshold = 2, expand_quality = 0.25,
        predictor = :secant, tracking_maxiters = 10, maxsteps = 10000
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
    if tracking_maxiters !== nothing && tracking_maxiters < 1
        throw(
            ArgumentError(
                "HomotopySweep `tracking_maxiters` must be ≥ 1 (nothing disables the " *
                    "cap), got $tracking_maxiters"
            )
        )
    end
    if maxsteps < 1
        throw(ArgumentError("HomotopySweep `maxsteps` must be ≥ 1, got $maxsteps"))
    end
    return HomotopySweep(
        inner, nsteps, adaptive, initial_step_factor, min_dλ,
        max_step_factor, expand_factor, expand_threshold, expand_quality, predictor,
        tracking_maxiters, maxsteps
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

# Resolve the iteration budget the inner corrector runs under for interior tracking
# steps, returning `(budget, cap_active)`. An explicit user `maxiters` always wins
# (solve kwargs shadow problem kwargs, matching the splat order of the inner solves),
# so the tracking cap is only active when the user passed none and `tracking_maxiters`
# is set. The 1000 fallback is the inner solvers' own `maxiters` default.
function _tracking_budget(tracking_maxiters, prob_kwargs, kwargs)
    haskey(kwargs, :maxiters) && return Int(kwargs[:maxiters]), false
    haskey(prob_kwargs, :maxiters) && return Int(prob_kwargs[:maxiters]), false
    tracking_maxiters === nothing && return 1000, false
    return Int(tracking_maxiters), true
end

# AUTO-07p ADPTDS-style effort bands, scaling the growth multiplier by the accepted
# step's corrector iteration count `nit` relative to its iteration budget: a near-free
# corrector earns the full `expand_factor`, moderate effort earns milder growth, and
# effort past a quarter of the budget holds the step — growing that close to the
# struggle zone mostly buys the next rejection. `nit < 0` encodes "no stats available"
# and keeps the classic unscaled behavior. The absolute floors keep the bands
# meaningful when the budget is hand-tuned small (`tracking_maxiters` of 10–20).
function _effort_growth_factor(nit::Int, budget::Int, expand_factor::T) where {T}
    nit < 0 && return expand_factor
    nit <= max(3, budget ÷ 20) && return expand_factor
    nit <= max(6, budget ÷ 4) && return one(T) + (expand_factor - one(T)) / 2
    return one(T)
end

# A success whose corrector consumed at least 3/4 of its iteration budget is the
# earliest warning that the next step will be rejected (AUTO's NIT ≥ ITNW band): the
# caller shrinks the step proactively instead of paying for a full-cost failure. The
# signal is only trusted when `nit ≤ budget`: a polyalgorithm inner reports `nsteps`
# aggregated across ALL its ladder members, so an over-budget count means an early
# member burned its budget before a later one succeeded cheaply — not that the
# successful corrector was struggling. Misreading that as struggle collapses the
# increment on every accepted step (measured: a 400× blowup on an n = 50 system), so
# an untrustworthy count may withhold growth but never shrink.
function _effort_wants_shrink(nit::Int, budget::Int)
    return nit >= 0 && nit <= budget && 4 * nit >= 3 * budget
end

# Full-budget standalone solve at a fixed λ. Used only when the tracking cap is active
# but must not bind — rescuing the λspan[1] anchor and the final landing on λspan[2] —
# so it runs with the user's original kwargs, outside the capped cache.
function _sweep_uncapped_solve(
        prob::SciMLBase.HomotopyProblem{uType, iip}, inner, uguess, λfix,
        args...; kwargs...
    ) where {uType, iip}
    fλ = SciMLBase.NonlinearFunction{iip}(FixLambda(prob.f, λfix))
    inner_prob = NonlinearProblem{iip}(fλ, copy(uguess), prob.p)
    return solve(inner_prob, inner, args...; prob.kwargs..., kwargs...)
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
    budget, cap_active = _tracking_budget(alg.tracking_maxiters, prob.kwargs, kwargs)

    # One inner-solver cache, built once around the mutable `FixLambda` and reused for
    # every solve of the sweep (the anchor below and each continuation step): advancing
    # λ is a field write plus a `reinit!` with the new warm start, so the inner solver's
    # workspace is reused instead of reconstructed. `guess` is handed to the cache and
    # may be iterated in place when aliasing is forwarded; `u` is a separately-owned
    # buffer the inner solver never writes, so a FAILED attempt leaves the last
    # converged iterate intact for retries and the failure returns below. When the
    # tracking cap is active it is baked into this cache (the many interior steps all
    # run capped); the anchor and the final landing are exempted through the
    # `_sweep_uncapped_solve` rescues below rather than a second full-budget cache, so
    # the happy path keeps a single inner workspace.
    fixλ = FixLambda(prob.f, λ)
    fλ = SciMLBase.NonlinearFunction{iip}(fixλ)
    guess = copy(u)
    cache = if cap_active
        init(
            NonlinearProblem{iip}(fλ, guess, prob.p), alg.inner, args...;
            prob.kwargs..., kwargs..., maxiters = budget
        )
    else
        init(
            NonlinearProblem{iip}(fλ, guess, prob.p), alg.inner, args...;
            prob.kwargs..., kwargs...
        )
    end

    # Anchor: solve the system at λ = λspan[1] from u0 BEFORE stepping. For the
    # canonical (0, 1) span this is the pure `simplified` system — the one the
    # homotopy contract is designed to make solvable from a cold start (OMC's
    # reference implementation also solves λ = 0 first). Without this anchor the
    # first inner solve runs at λ0 + dλ warm-started from u0, so a poor u0 can
    # converge onto the wrong branch and the sweep then tracks that branch all
    # the way to a wrong root with a success retcode.
    last_sol = CommonSolve.solve!(cache)
    if cap_active && !SciMLBase.successful_retcode(last_sol)
        # The anchor is exempt from the tracking cap — it is the one cold-start solve
        # the homotopy contract is designed around, so it may legitimately need many
        # more iterations than a warm-started tracking step. Rerun it with the full
        # user budget before declaring the homotopy premise broken.
        last_sol = _sweep_uncapped_solve(prob, alg.inner, u, λ, args...; kwargs...)
    end
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
    attempts = 0

    while true
        attempts += 1
        if attempts > alg.maxsteps
            # Cap on total attempts (accepted steps plus bisection retries), mirroring
            # ArcLengthContinuation's guard: without it the loop is bounded only by
            # span/min_dλ ≈ 7×10⁷ accepted steps.
            return SciMLBase.build_solution(
                prob, alg, u, nothing; retcode = ReturnCode.MaxIters
            )
        end
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

        if cap_active && next_λ == λend && !SciMLBase.successful_retcode(last_sol)
            # The final landing on λspan[2] is exempt from the tracking cap: give it
            # the full user budget before letting the failure feed the bisection
            # logic. The prediction is recomputed (never reuse `guess`: the capped
            # solve may have iterated in place in that buffer).
            retry_guess = if used_secant
                _sweep_extrapolate!(virtual, u, u_prev, (next_λ - λ) / (λ - λ_prev))
            else
                u
            end
            last_sol = _sweep_uncapped_solve(
                prob, alg.inner, retry_guess, next_λ, args...; kwargs...
            )
        end

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
                nit = last_sol.stats === nothing ? -1 : Int(last_sol.stats.nsteps)
                if _effort_wants_shrink(nit, budget)
                    # Proactive shrink on a straining success (see
                    # `_effort_wants_shrink`); the floor guard keeps the increment
                    # from dropping below what bisection itself may reach.
                    abs(dλ) / 2 >= min_dλ && (dλ = dλ / 2)
                    streak = 0
                else
                    streak += 1
                    # Growth requires a streak of successes (the classic heuristic)
                    # plus evidence the corrector has headroom: either a small relative
                    # prediction error (the quality gate) or a corrector that converged
                    # almost immediately. The iteration count covers paths that flatten
                    # exponentially, where θ stays at a constant mediocre value while
                    # the absolute corrections — and hence the corrector work — become
                    # negligible. The growth factor itself is scaled by the corrector's
                    # effort (see `_effort_growth_factor`), so the gates veto growth
                    # and the effort bands size it. The streak is NOT reset on a vetoed
                    # or held expansion, so growth resumes on the first step whose
                    # evidence recovers.
                    corrector_cheap = nit >= 0 && nit <= 2
                    if streak >= alg.expand_threshold &&
                            (θ === nothing || θ <= λT(alg.expand_quality) || corrector_cheap)
                        g = _effort_growth_factor(nit, budget, expand_factor)
                        if g > 1
                            grown = g * dλ
                            dλ = abs(grown) > abs(max_dλ) ? max_dλ : grown
                            streak = 0
                        end
                    end
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
