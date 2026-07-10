"""
    SimpleHomotopySweep(; inner = SimpleNewtonRaphson(), nsteps = nothing,
        adaptive = true, initial_step_factor = 0.1, min_dλ = nothing,
        max_step_factor = 1.0, expand_factor = 2.0, expand_threshold = 2,
        expand_quality = 0.25, predictor = :secant, tracking_maxiters = 10,
        tracking_abstol = nothing, maxsteps = 10000)

Natural-parameter continuation solver for a `SciMLBase.HomotopyProblem`, the
SimpleNonlinearSolve counterpart of `HomotopySweep`. The algorithm is the same —
anchor solve at `λspan[1]`, then predictor-corrector λ-stepping with the classic
success/failure step control (failure halves the increment; `expand_threshold`
consecutive successes grow it by `expand_factor` up to `max_step_factor` of the span,
gated on the `expand_quality` secant-prediction error estimate), with a
trust-monitored `:secant` warm-start predictor (`:constant` disables extrapolation) —
but the driver is written in the direct, value-oriented SimpleNonlinearSolve style:
each step calls `solve` on a freshly constructed (stack-allocated) inner problem
instead of maintaining an inner-solver cache, and all sweep state is plain values.

Optional derivative fields of the problem's `NonlinearFunction` (which
`SciMLBase.HomotopyProblem` requires to follow the same λ-extended argument convention
as the residual) are consumed by this solver: an analytic `jac(u, p, λ)` /
`jac(J, u, p, λ)` is λ-fixed exactly like the residual and handed to the inner solver
as a standard 2/3-argument Jacobian, and `jac_prototype`, `sparsity`, and `colorvec`
are forwarded unchanged. The prototype is not eltype-promoted with λ: supply a
prototype whose eltype matches the promoted residual eltype if λ's precision differs
from `u0`'s.

With a `StaticArray` (or scalar) `u0` and a SimpleNonlinearSolve inner solver, the
entire sweep is non-allocating, which also makes it the variant of choice inside hot
loops, on GPUs, and for compilation-sensitive targets. For large mutable systems,
prefer `HomotopySweep`, whose cached driver reuses the inner solver's workspace
across steps.

Keyword arguments are identical to `HomotopySweep` except that `inner` defaults to
`SimpleNewtonRaphson()` (a SimpleNonlinearSolve corrector) rather than the
NonlinearSolve polyalgorithm. In particular `tracking_maxiters` caps the inner
corrector's iterations for interior tracking steps only (never the `λspan[1]` anchor,
never the final landing on `λspan[2]`, and never overriding an explicit user-passed
`maxiters`), `tracking_abstol` (default `nothing` = disabled) loosens the inner
corrector's absolute tolerance on those same interior steps only — the anchor and the
final landing always run at the user's full tolerances (each step here is a fresh
standalone solve, so the exemption is simply not splicing the loose tolerance into
the exempt solves — no re-polish is needed), and an explicit user-passed `abstol` or
`reltol` (solve kwarg or problem kwarg) disables the loosening entirely — the
success-side step growth is scaled by the corrector's iteration count (the AUTO-07p
`ADPTDS` effort bands, including a proactive halving when a success nearly exhausts
the budget), and `maxsteps` caps the total number of predictor-corrector attempts,
returning `ReturnCode.MaxIters` with the last converged iterate when exceeded.

When the sweep cannot reach the end of `λspan`, the returned solution carries a
failure retcode: its `u` is the last converged iterate (at some ``λ`` short of
`λspan[2]`, or `u0` itself if the initial `λspan[1]` anchor solve failed), while
`resid` and `original` come from the most recent inner solve (unlike `HomotopySweep`,
the `ReturnCode.Stalled` path reports the residual of the last *successful* step
rather than `nothing`: every return path builds the same concrete solution type,
which is part of what keeps the sweep allocation-free).
"""
@concrete struct SimpleHomotopySweep <: AbstractNonlinearSolveAlgorithm
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
    tracking_abstol
    maxsteps::Int
end

function SimpleHomotopySweep(;
        inner = SimpleNewtonRaphson(), nsteps = nothing, adaptive = true,
        initial_step_factor = 0.1, min_dλ = nothing, max_step_factor = 1.0,
        expand_factor = 2.0, expand_threshold = 2, expand_quality = 0.25,
        predictor = :secant, tracking_maxiters = 10, tracking_abstol = nothing,
        maxsteps = 10000
    )
    if nsteps !== nothing && nsteps < 1
        throw(ArgumentError("SimpleHomotopySweep `nsteps` must be ≥ 1, got $nsteps"))
    end
    if !adaptive && nsteps === nothing
        throw(
            ArgumentError(
                "SimpleHomotopySweep with `adaptive = false` takes fixed-size λ steps, " *
                    "so an explicit `nsteps` is required."
            )
        )
    end
    if !(0 < initial_step_factor <= 1)
        throw(
            ArgumentError(
                "SimpleHomotopySweep `initial_step_factor` must be in (0, 1], got $initial_step_factor"
            )
        )
    end
    if min_dλ !== nothing && min_dλ <= 0
        throw(
            ArgumentError(
                "SimpleHomotopySweep `min_dλ` must be positive, got $min_dλ"
            )
        )
    end
    if !(0 < max_step_factor <= 1)
        throw(
            ArgumentError(
                "SimpleHomotopySweep `max_step_factor` must be in (0, 1], got $max_step_factor"
            )
        )
    end
    if expand_factor < 1
        throw(
            ArgumentError(
                "SimpleHomotopySweep `expand_factor` must be ≥ 1 (1 disables " *
                    "expansion), got $expand_factor"
            )
        )
    end
    if expand_threshold < 1
        throw(
            ArgumentError(
                "SimpleHomotopySweep `expand_threshold` must be ≥ 1, got $expand_threshold"
            )
        )
    end
    if !(expand_quality > 0)
        throw(
            ArgumentError(
                "SimpleHomotopySweep `expand_quality` must be positive (Inf disables " *
                    "the gate), got $expand_quality"
            )
        )
    end
    if predictor !== :secant && predictor !== :constant
        throw(
            ArgumentError(
                "SimpleHomotopySweep `predictor` must be :secant or :constant, got :$predictor"
            )
        )
    end
    if tracking_maxiters !== nothing && tracking_maxiters < 1
        throw(
            ArgumentError(
                "SimpleHomotopySweep `tracking_maxiters` must be ≥ 1 (nothing " *
                    "disables the cap), got $tracking_maxiters"
            )
        )
    end
    if tracking_abstol !== nothing && !(tracking_abstol > 0)
        throw(
            ArgumentError(
                "SimpleHomotopySweep `tracking_abstol` must be positive (nothing " *
                    "disables the loose tracking tolerance), got $tracking_abstol"
            )
        )
    end
    if maxsteps < 1
        throw(ArgumentError("SimpleHomotopySweep `maxsteps` must be ≥ 1, got $maxsteps"))
    end
    return SimpleHomotopySweep(
        inner, nsteps, adaptive, initial_step_factor, min_dλ,
        max_step_factor, expand_factor, expand_threshold, expand_quality, predictor,
        tracking_maxiters, tracking_abstol, maxsteps
    )
end

# Immutable λ-fixing wrapper (contrast NonlinearSolveBase's mutable `FixLambda`): the
# sweep rebuilds it each step, which for isbits `f` is stack-allocated — the immutable
# form is what keeps the StaticArray path heap-free.
struct SimpleFixLambda{F, T}
    f::F
    λ::T
end
(fl::SimpleFixLambda)(args...) = fl.f(args..., fl.λ)

# Builds the inner NonlinearFunction for one step, forwarding the derivative fields of
# the problem's NonlinearFunction: `jac` is λ-fixed through a second `SimpleFixLambda`
# constructed here from the same `λfix` value as the residual's (both wrappers are
# immutable and rebuilt together each step, so they cannot desynchronize);
# `jac_prototype`/`sparsity`/`colorvec` pass through unchanged (the prototype is NOT
# eltype-promoted with λ, so its eltype should match the promoted residual eltype when
# λ's precision differs from `u0`'s). When no derivative field is present the
# construction is exactly the bare positional one — the branch is decided by the
# problem's type, so the unused arm is compiled away and the StaticArray/scalar path
# stays allocation-free.
function _simple_sweep_nonlinear_function(::Val{iip}, f, λfix) where {iip}
    if f.jac === nothing && f.jac_prototype === nothing && f.sparsity === nothing &&
            f.colorvec === nothing
        return NonlinearFunction{iip}(SimpleFixLambda(f, λfix))
    end
    jac = f.jac === nothing ? nothing : SimpleFixLambda(f.jac, λfix)
    return NonlinearFunction{iip}(
        SimpleFixLambda(f, λfix); jac, jac_prototype = f.jac_prototype,
        sparsity = f.sparsity, colorvec = f.colorvec
    )
end

# One inner corrector solve at fixed λ. A fresh immutable problem per call: for
# StaticArray/scalar `u` everything here lives on the stack.
function _simple_sweep_solve(
        prob::SciMLBase.HomotopyProblem{uType, iip}, inner, guess, λfix,
        args...; kwargs...
    ) where {uType, iip}
    fλ = _simple_sweep_nonlinear_function(Val(iip), prob.f, λfix)
    inner_prob = NonlinearProblem{iip}(fλ, guess, prob.p)
    return solve(inner_prob, inner, args...; prob.kwargs..., kwargs...)
end

# The corrector may iterate in place in its u0 for mutable arrays, so hand it a copy;
# immutable/StaticArray/scalar `u` cannot be mutated and passes through (stack).
_simple_sweep_guess(u) = NLBUtils.can_setindex(u) ? copy(u) : u

# Local duplicates of NonlinearSolveBase's `_tracking_budget` / `_tracking_tolerance`
# / `_effort_growth_factor` / `_effort_wants_shrink` (see homotopy_sweep.jl there for
# the rationale): duplicated rather than shared so this package does not depend on new
# NonlinearSolveBase internals, and kept branch-only/isbits so the StaticArray path
# stays kernel-safe.
function _simple_tracking_budget(tracking_maxiters, prob_kwargs, kwargs)
    haskey(kwargs, :maxiters) && return Int(kwargs[:maxiters]), false
    haskey(prob_kwargs, :maxiters) && return Int(prob_kwargs[:maxiters]), false
    tracking_maxiters === nothing && return 1000, false
    return Int(tracking_maxiters), true
end

# any explicit user tolerance (abstol OR reltol; solve kwargs shadow problem kwargs)
# disables the loosening — a loose abstol spliced next to a user reltol would fire
# first in the default OR-combined termination modes
function _simple_tracking_tolerance(tracking_abstol, prob_kwargs, kwargs)
    tracking_abstol === nothing && return nothing, false
    (haskey(kwargs, :abstol) || haskey(kwargs, :reltol)) && return nothing, false
    (haskey(prob_kwargs, :abstol) || haskey(prob_kwargs, :reltol)) &&
        return nothing, false
    return tracking_abstol, true
end

function _simple_effort_growth_factor(nit::Int, budget::Int, expand_factor::T) where {T}
    nit < 0 && return expand_factor
    nit <= max(3, budget ÷ 20) && return expand_factor
    nit <= max(6, budget ÷ 4) && return one(T) + (expand_factor - one(T)) / 2
    return one(T)
end

# only trusted when nit ≤ budget: a polyalgorithm inner aggregates nsteps across its
# ladder, and misreading that as corrector struggle would collapse the increment (see
# NonlinearSolveBase._effort_wants_shrink) — an untrustworthy count holds, never shrinks
function _simple_effort_wants_shrink(nit::Int, budget::Int)
    return nit >= 0 && nit <= budget && 4 * nit >= 3 * budget
end

function CommonSolve.solve(
        prob::SciMLBase.HomotopyProblem{uType, iip},
        alg::SimpleHomotopySweep, args...; kwargs...
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
    u = _simple_sweep_guess(prob.u0)
    budget, cap_active = _simple_tracking_budget(alg.tracking_maxiters, prob.kwargs, kwargs)
    track_abstol, tol_active = _simple_tracking_tolerance(
        alg.tracking_abstol, prob.kwargs, kwargs
    )

    # Anchor: solve the system at λ = λspan[1] from u0 before stepping (see
    # `HomotopySweep` — same contract: a failed anchor means the homotopy premise is
    # broken and no continuation is attempted).
    last_sol = _simple_sweep_solve(
        prob, alg.inner, _simple_sweep_guess(u), λ, args...; kwargs...
    )
    if !SciMLBase.successful_retcode(last_sol)
        return SciMLBase.build_solution(
            prob, alg, u, last_sol.resid;
            retcode = last_sol.retcode, original = last_sol
        )
    end
    u = _simple_sweep_guess(last_sol.u)
    # Zero-width λspan (λ0 == λend): the anchor IS the single target solve.
    λ == λend && return SciMLBase.build_solution(
        prob, alg, u, last_sol.resid; retcode = ReturnCode.Success, original = last_sol
    )

    # λ_prev == λ means there is no secant history yet (constant warm start). The
    # trust counter and quality gate mirror `HomotopySweep` exactly; see its docstring
    # for the controller's rationale.
    u_prev = u
    λ_prev = λ
    streak = 0
    trust = 2
    disp_prev = zero(λT)
    attempts = 0

    while true
        attempts += 1
        if attempts > alg.maxsteps
            # Cap on total attempts (accepted steps plus bisection retries); the same
            # concrete solution type as every other return path (see the Stalled path).
            return SciMLBase.build_solution(
                prob, alg, u, last_sol.resid;
                retcode = ReturnCode.MaxIters, original = last_sol
            )
        end
        next_λ = abs(λend - λ) <= abs(dλ) ? λend : λ + dλ
        if next_λ == λ && next_λ != λend
            # dλ underflowed below eps(λ) mid-continuation: no further progress.
            # `resid`/`original` come from the most recent inner solve (NOT `nothing`
            # as in `HomotopySweep`): every return path builds the same concrete
            # `NonlinearSolution` type, which is what lets the whole sweep stay
            # allocation-free on StaticArrays.
            return SciMLBase.build_solution(
                prob, alg, u, last_sol.resid;
                retcode = ReturnCode.Stalled, original = last_sol
            )
        end
        used_secant = alg.predictor === :secant && trust >= 2 && λ_prev != λ
        guess = if used_secant
            s = (next_λ - λ) / (λ - λ_prev)
            @. u + s * (u - u_prev)
        else
            _simple_sweep_guess(u)
        end
        # Interior tracking steps run under the tracking cap and the loose tracking
        # tolerance; the final landing on λend keeps the full user budget and full
        # tolerances (the anchor above never enters this loop) — every step here is a
        # fresh standalone solve, so the exemption is simply not splicing the loose
        # kwargs, with no re-polish needed. The cap/tolerance are spliced BEFORE the
        # user kwargs so explicit user values always win (and `cap_active`/`tol_active`
        # are false in those cases anyway).
        interior = next_λ != λend
        last_sol = if interior && cap_active && tol_active
            _simple_sweep_solve(
                prob, alg.inner, guess, next_λ, args...;
                maxiters = budget, abstol = track_abstol, kwargs...
            )
        elseif interior && cap_active
            _simple_sweep_solve(
                prob, alg.inner, guess, next_λ, args...; maxiters = budget, kwargs...
            )
        elseif interior && tol_active
            _simple_sweep_solve(
                prob, alg.inner, guess, next_λ, args...;
                abstol = track_abstol, kwargs...
            )
        else
            _simple_sweep_solve(prob, alg.inner, guess, next_λ, args...; kwargs...)
        end

        if SciMLBase.successful_retcode(last_sol)
            θ = nothing
            if λ_prev != λ
                # measured against the prediction the secant WOULD have made, so trust
                # can recover on constant-warm-started steps too
                sv = (next_λ - λ) / (λ - λ_prev)
                virtual = @. u + sv * (u - u_prev)
                correction = L2_NORM(last_sol.u .- virtual)
                disp = L2_NORM(last_sol.u .- u)
                scale = max(disp, disp_prev, sqrt(eps(λT)) * (1 + L2_NORM(last_sol.u)))
                θ = correction / scale
                trust = θ < 1 / 2 ? trust + 1 : 0
                disp_prev = disp
            else
                disp_prev = L2_NORM(last_sol.u .- u)
            end
            u_prev = u
            λ_prev = λ
            u = _simple_sweep_guess(last_sol.u)
            λ = next_λ
            λ == λend && break
            if alg.adaptive
                nit = last_sol.stats === nothing ? -1 : Int(last_sol.stats.nsteps)
                if _simple_effort_wants_shrink(nit, budget)
                    # proactive shrink on a straining success (AUTO's NIT ≥ ITNW band)
                    abs(dλ) / 2 >= min_dλ && (dλ = dλ / 2)
                    streak = 0
                else
                    streak += 1
                    corrector_cheap = nit >= 0 && nit <= 2
                    if streak >= alg.expand_threshold &&
                            (θ === nothing || θ <= λT(alg.expand_quality) || corrector_cheap)
                        g = _simple_effort_growth_factor(nit, budget, expand_factor)
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
            trust = 0
        else
            return SciMLBase.build_solution(
                prob, alg, u, last_sol.resid;
                retcode = last_sol.retcode, original = last_sol
            )
        end
    end

    return SciMLBase.build_solution(
        prob, alg, u, last_sol.resid; retcode = ReturnCode.Success, original = last_sol
    )
end
