const RelNormModes = Union{
    RelNormTerminationMode, RelNormSafeTerminationMode, RelNormSafeBestTerminationMode,
}
const AbsNormModes = Union{
    AbsNormTerminationMode, AbsNormSafeTerminationMode, AbsNormSafeBestTerminationMode,
}
# The four modes carrying a `gtol` field; the plain `*NormTerminationMode`s do not.
const SafeNormModes = Union{
    RelNormSafeTerminationMode, RelNormSafeBestTerminationMode,
    AbsNormSafeTerminationMode, AbsNormSafeBestTerminationMode,
}

"""
    residual_only_termination_mode(mode) -> Bool

Return whether `mode` decides termination from the residual alone, without reading the
iterate or displacement from the previous iterate and without retaining per-step history.

This developer trait is used by [`supports_deferred_residual`](@ref) to determine whether a
solver may honor `evaluate_residual = false`. A deferred step reports no displacement and
reaches the termination check only when the driver requests the residual, so a mode that
reads displacement or retains step history would observe a step that did not occur.

This is a developer API for packages implementing a
[`AbstractNonlinearTerminationMode`](@ref), not a user-facing solver option. A custom mode
may return `true` only when its convergence decision depends on the current residual and
tolerances, not on the iterate, displacement, or per-step history.

# Arguments

- `mode::AbstractNonlinearTerminationMode`: The termination mode to inspect.

# Returns

`true` for residual-only modes and `false` for modes that inspect displacement or retain
per-step state. The default implementation returns `false`; solver packages should add a
method for a new termination mode only when it satisfies the residual-only contract.

# Examples

```julia
using NonlinearSolveBase

NonlinearSolveBase.residual_only_termination_mode(AbsTerminationMode()) # true
NonlinearSolveBase.residual_only_termination_mode(RelTerminationMode()) # false
```
"""
residual_only_termination_mode(::AbstractNonlinearTerminationMode) = false
residual_only_termination_mode(::AbsTerminationMode) = true
residual_only_termination_mode(::AbsNormTerminationMode) = true

# Core Implementation
@concrete mutable struct NonlinearTerminationModeCache{uType, T}
    u::uType
    retcode::ReturnCode.T
    abstol::T
    reltol::T
    best_objective_value::T
    mode
    initial_objective
    objectives_trace
    nsteps::Int
    saved_values
    u0_norm
    step_norm_trace
    max_stalled_steps
    # Scratch for `u - uprev`. It is sized from `u`, whereas `uType` is the type of the
    # retained best iterate, which is `nothing` for every non-`Best` mode.
    u_diff_cache
    leastsq::Bool
end

get_abstol(cache::NonlinearTerminationModeCache) = cache.abstol
get_reltol(cache::NonlinearTerminationModeCache) = cache.reltol

function termination_condition_result(cache::NonlinearTerminationModeCache, args...)
    return termination_condition_result(cache.mode, cache, args...)
end

function termination_condition_result(
        ::AbstractNonlinearTerminationMode, cache, fallback_u, fallback_t, solver_retcode
    )
    retcode = ifelse(
        solver_retcode == ReturnCode.Terminated, ReturnCode.Success, ReturnCode.Failure
    )
    return fallback_u, fallback_t, retcode
end

function termination_condition_result(
        ::AbstractSafeNonlinearTerminationMode, cache, fallback_u, fallback_t, solver_retcode
    )
    retcode = if solver_retcode == ReturnCode.Terminated
        ifelse(cache.retcode != ReturnCode.Default, cache.retcode, ReturnCode.Success)
    elseif solver_retcode == ReturnCode.Success
        ReturnCode.Failure
    else
        solver_retcode
    end
    return fallback_u, fallback_t, retcode
end

function termination_condition_result(
        ::AbstractSafeBestNonlinearTerminationMode, cache, fallback_u, fallback_t,
        solver_retcode
    )
    u = cache.u === nothing ? fallback_u : copy(cache.u)
    t = cache.saved_values === nothing ? fallback_t : only(cache.saved_values)
    retcode = if solver_retcode == ReturnCode.Terminated
        ifelse(cache.retcode != ReturnCode.Default, cache.retcode, ReturnCode.Success)
    elseif solver_retcode == ReturnCode.Success
        ReturnCode.Failure
    else
        solver_retcode
    end
    return u, t, retcode
end

function update_u!!(cache::NonlinearTerminationModeCache, u)
    cache.u === nothing && return
    return if cache.u isa AbstractArray && ArrayInterface.can_setindex(cache.u)
        copyto!(cache.u, u)
    else
        cache.u = u
    end
end

"""
    alloc_u_diff_cache(u)

Scratch to hold `u - uprev` for the stall test. Sized from `u`, since the retained best
iterate a `Best` mode carries is absent from every other mode. When `u` cannot be written
into in place the difference is rebuilt each step, so this only has to fix the type.
"""
function alloc_u_diff_cache(u)
    (u isa Number || !ArrayInterface.can_setindex(u)) && return u .- u
    return similar(u)
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, mode::AbstractNonlinearTerminationMode, du, u,
        saved_value_prototype...; abstol = nothing, reltol = nothing, kwargs...
    )
    T = promote_type(eltype(du), eltype(u))
    abstol = get_tolerance(u, abstol, T)
    reltol = get_tolerance(u, reltol, T)
    TT = typeof(abstol)

    u_unaliased = mode isa AbstractSafeBestNonlinearTerminationMode ?
        (ArrayInterface.can_setindex(u) ? copy(u) : u) : nothing

    if mode isa AbstractSafeNonlinearTerminationMode
        if mode isa AbsNormSafeTerminationMode || mode isa AbsNormSafeBestTerminationMode
            initial_objective = Utils.apply_norm(mode.internalnorm, du)
            u0_norm = nothing
        else
            initial_objective = Utils.apply_norm(mode.internalnorm, du) /
                (Utils.apply_norm(mode.internalnorm, du, u) + eps(TT))
            u0_norm = mode.max_stalled_steps === nothing ? nothing : L2_NORM(u)
        end
        objectives_trace = Vector{TT}(undef, mode.patience_steps)
        step_norm_trace = mode.max_stalled_steps === nothing ? nothing :
            Vector{TT}(undef, mode.max_stalled_steps)
        u_diff_cache = step_norm_trace === nothing ? nothing : alloc_u_diff_cache(u)
        best_value = initial_objective
        max_stalled_steps = mode.max_stalled_steps
    else
        initial_objective = nothing
        objectives_trace = nothing
        u0_norm = nothing
        step_norm_trace = nothing
        best_value = Utils.convert_real(T, Inf)
        max_stalled_steps = nothing
        u_diff_cache = nothing
    end

    length(saved_value_prototype) == 0 && (saved_value_prototype = nothing)

    leastsq = typeof(prob) <: NonlinearLeastSquaresProblem
    return NonlinearTerminationModeCache(
        u_unaliased, ReturnCode.Default, abstol, reltol, best_value, mode,
        initial_objective, objectives_trace, 0, saved_value_prototype,
        u0_norm, step_norm_trace, max_stalled_steps, u_diff_cache, leastsq
    )
end

function SciMLBase.reinit!(
        cache::NonlinearTerminationModeCache, du, u, saved_value_prototype...;
        abstol = cache.abstol, reltol = cache.reltol, kwargs...
    )
    T = eltype(cache.abstol)
    length(saved_value_prototype) != 0 && (cache.saved_values = saved_value_prototype)

    mode = cache.mode
    if cache.u !== nothing
        if u isa Number || !ArrayInterface.can_setindex(u)
            cache.u = u
        else
            cache.u .= u
        end
    end
    cache.retcode = ReturnCode.Default

    cache.abstol = get_tolerance(u, abstol, T)
    cache.reltol = get_tolerance(u, reltol, T)
    cache.nsteps = 0
    TT = typeof(cache.abstol)

    return if mode isa AbstractSafeNonlinearTerminationMode
        if mode isa AbsNormSafeTerminationMode || mode isa AbsNormSafeBestTerminationMode
            cache.initial_objective = Utils.apply_norm(mode.internalnorm, du)
        else
            cache.initial_objective = Utils.apply_norm(mode.internalnorm, du) /
                (Utils.apply_norm(mode.internalnorm, du, u) + eps(TT))
            cache.max_stalled_steps !== nothing && (cache.u0_norm = L2_NORM(u))
        end
        cache.best_objective_value = cache.initial_objective
    else
        cache.best_objective_value = Utils.convert_real(T, Inf)
    end
end

## This dispatch is needed based on how Terminating Callback works!
function (cache::NonlinearTerminationModeCache)(
        integrator::AbstractODEIntegrator, abstol::Number, reltol::Number, min_t
    )
    if min_t === nothing || integrator.t ≥ min_t
        return cache(
            cache.mode, SciMLBase.get_du(integrator),
            integrator.u, integrator.uprev, abstol, reltol
        )
    end
    return false
end
function (cache::NonlinearTerminationModeCache)(du, u, uprev, args...)
    return cache(cache.mode, du, u, uprev, cache.abstol, cache.reltol, args...)
end

function (cache::NonlinearTerminationModeCache)(
        mode::AbstractNonlinearTerminationMode, du, u, uprev, abstol, reltol, args...
    )
    if check_convergence(mode, du, u, uprev, abstol, reltol)
        cache.retcode = ReturnCode.Success
        return true
    end
    return false
end

function (cache::NonlinearTerminationModeCache)(
        mode::AbstractSafeNonlinearTerminationMode, du, u, uprev, abstol, reltol, args...
    )
    if mode isa AbsNormSafeTerminationMode || mode isa AbsNormSafeBestTerminationMode
        objective = Utils.apply_norm(mode.internalnorm, du)
        criteria = abstol
    else
        objective = Utils.apply_norm(mode.internalnorm, du) /
            (Utils.apply_norm(mode.internalnorm, du, u) + eps(reltol))
        criteria = reltol
    end

    # Protective Break
    if !isfinite(objective)
        cache.retcode = ReturnCode.Unstable
        return true
    end

    # By default we turn this off since it have potential for false positives
    if mode.protective_threshold !== nothing &&
            (objective > cache.initial_objective * mode.protective_threshold * length(du))
        cache.retcode = ReturnCode.Unstable
        return true
    end

    # Check if it is the best solution
    if mode isa AbstractSafeBestNonlinearTerminationMode &&
            objective < cache.best_objective_value
        cache.best_objective_value = objective
        update_u!!(cache, u)
        cache.saved_values !== nothing && length(args) ≥ 1 && (cache.saved_values = args)
    end

    # Main Termination Criteria
    if objective ≤ criteria
        cache.retcode = ReturnCode.Success
        return true
    end

    # Terminate if we haven't improved for the last `patience_steps`
    cache.nsteps += 1
    cache.objectives_trace[mod1(cache.nsteps, length(cache.objectives_trace))] = objective

    if objective ≤ mode.patience_objective_multiplier * criteria &&
            cache.nsteps > mode.patience_steps
        if cache.nsteps < length(cache.objectives_trace)
            min_obj, max_obj = extrema(@view(cache.objectives_trace[1:(cache.nsteps)]))
        else
            min_obj, max_obj = extrema(cache.objectives_trace)
        end
        if min_obj < mode.min_max_factor * max_obj
            if cache.leastsq
                # If least squares, found a local minima thus success
                cache.retcode = ReturnCode.StalledSuccess
            else
                # Not a success if f(x)>0 and residual too high
                cache.retcode = ReturnCode.Stalled
            end
            return true
        end
    end

    # Test for stalling if that is enabled
    if cache.step_norm_trace !== nothing
        if cache.u_diff_cache !== nothing && !(u isa Number) &&
                ArrayInterface.can_setindex(cache.u_diff_cache)
            @. cache.u_diff_cache = u - uprev
        else
            cache.u_diff_cache = u .- uprev
        end
        du_norm = L2_NORM(cache.u_diff_cache)
        cache.step_norm_trace[mod1(cache.nsteps, length(cache.step_norm_trace))] = du_norm
        if cache.nsteps > mode.max_stalled_steps
            max_step_norm = maximum(cache.step_norm_trace)
            if mode isa AbsNormSafeTerminationMode ||
                    mode isa AbsNormSafeBestTerminationMode
                stalled_step = max_step_norm ≤ abstol
            else
                stalled_step = max_step_norm ≤ reltol * (max_step_norm + cache.u0_norm)
            end
            if stalled_step
                if cache.leastsq
                    cache.retcode = ReturnCode.StalledSuccess
                else
                    cache.retcode = ReturnCode.Stalled
                end
                return true
            end
        end
    end

    cache.retcode = ReturnCode.Failure
    return false
end

# Check Convergence
function check_convergence(::RelTerminationMode, duₙ, uₙ, __, ___, reltol)
    if Utils.fast_scalar_indexing(duₙ)
        return all(
            @closure(
                xy -> begin
                    x, y = xy
                    return abs(y) ≤ reltol * abs(x + y)
                end
            ), zip(uₙ, duₙ)
        )
    else # using mapreduce here will almost certainly be faster on GPUs
        return mapreduce(
            @closure((xᵢ, yᵢ) -> (abs(yᵢ) ≤ reltol * abs(xᵢ + yᵢ))), *, uₙ, duₙ; init = true
        )
    end
end
function check_convergence(::AbsTerminationMode, duₙ, _, __, abstol, ___)
    return all(@closure(x -> abs(x) ≤ abstol), duₙ)
end

function check_convergence(norm::NormTerminationMode, duₙ, uₙ, _, abstol, reltol)
    du_norm = Utils.apply_norm(norm.internalnorm, duₙ)
    return (du_norm ≤ abstol) ||
        (du_norm ≤ reltol * Utils.apply_norm(norm.internalnorm, duₙ, uₙ))
end

function check_convergence(mode::RelNormModes, duₙ, uₙ, _, __, reltol)
    du_norm = Utils.apply_norm(mode.internalnorm, duₙ)
    return du_norm ≤ reltol * Utils.apply_norm(mode.internalnorm, duₙ, uₙ)
end

function check_convergence(mode::AbsNormModes, duₙ, _, __, abstol, ___)
    return Utils.apply_norm(mode.internalnorm, duₙ) ≤ abstol
end

# High-Level API with defaults.
## This is mostly for internal usage in NonlinearSolve and SimpleNonlinearSolve
function default_termination_mode(
        ::Union{ImmutableNonlinearProblem, NonlinearProblem}, ::Val{:simple}
    )
    return AbsNormTerminationMode(Base.Fix1(maximum, abs))
end
function default_termination_mode(::NonlinearLeastSquaresProblem, ::Val{:simple})
    return AbsNormTerminationMode(Base.Fix2(norm, 2))
end

function default_termination_mode(
        ::Union{ImmutableNonlinearProblem, NonlinearProblem}, ::Val{:regular}
    )
    return AbsNormSafeBestTerminationMode(Base.Fix1(maximum, abs); max_stalled_steps = 32)
end

function default_termination_mode(prob::NonlinearLeastSquaresProblem, ::Val{:regular})
    return AbsNormSafeBestTerminationMode(
        Base.Fix2(norm, 2); max_stalled_steps = 32,
        gtol = default_gradient_tolerance(eltype(prob.u0))
    )
end

function init_termination_cache(
        prob::AbstractNonlinearProblem, abstol, reltol, du, u, ::Nothing, callee::Val
    )
    return init_termination_cache(
        prob, abstol, reltol, du, u, default_termination_mode(prob, callee), callee
    )
end

function init_termination_cache(
        prob::AbstractNonlinearProblem, abstol, reltol, du,
        u, tc::AbstractNonlinearTerminationMode, ::Val
    )
    T = promote_type(eltype(du), eltype(u))
    abstol = get_tolerance(u, abstol, T)
    reltol = get_tolerance(u, reltol, T)
    cache = init(prob, tc, du, u; abstol, reltol)
    return abstol, reltol, cache
end

default_gradient_tolerance(::Type{T}) where {T} = sqrt(eps(float(real(one(T)))))

gradient_tolerance(::AbstractNonlinearTerminationMode) = nothing
gradient_tolerance(mode::SafeNormModes) = mode.gtol

gradient_measure_supported(_) = false
gradient_measure_supported(::AbstractMatrix) = true
gradient_measure_supported(::Number) = true

function gradient_stationarity_measure(J, fu)
    fu_norm = L2_NORM(fu)
    # `L2_NORM` is `@fastmath`, which lets LLVM assume its result is finite and fold an
    # `isfinite` test on it away. Non-finiteness has to be caught on the raw residual, and
    # every consumer of the measure must additionally fail closed.
    (iszero(fu_norm) || !_all_finite(fu)) && return nothing
    return _gradient_stationarity_measure(J, fu, fu_norm)
end

_all_finite(x::Number) = isfinite(x)
_all_finite(x) = all(isfinite, x)

function _gradient_stationarity_measure(J::Number, fu, fu_norm)
    iszero(J) && return nothing
    return abs(J' * fu) / (abs(J) * fu_norm)
end

function _gradient_stationarity_measure(J::AbstractMatrix, fu, fu_norm)
    fu_vec = Utils.safe_vec(fu)
    size(J, 1) == length(fu_vec) || return nothing
    g = J' * fu_vec
    measure = zero(fu_norm)
    for (j, gⱼ) in zip(axes(J, 2), g)
        col_norm = L2_NORM(@view(J[:, j]))
        iszero(col_norm) && continue
        measure = max(measure, abs(gⱼ) / (col_norm * fu_norm))
    end
    return measure
end

gradient_stationarity_retcode(leastsq::Bool) = ifelse(
    leastsq, ReturnCode.Success, ReturnCode.Stalled
)

# The objective a mode ranks iterates by. `Rel*` modes rank on a relative quantity, so
# comparing a bare residual norm against their `best_objective_value` mixes two scales.
mode_objective(mode, fu, u, reltol) = Utils.apply_norm(mode.internalnorm, fu)
function mode_objective(
        mode::Union{
            RelNormSafeTerminationMode, RelNormSafeBestTerminationMode,
        }, fu, u, reltol
    )
    return Utils.apply_norm(mode.internalnorm, fu) /
        (Utils.apply_norm(mode.internalnorm, fu, u) + eps(typeof(reltol)))
end

# A `Best` mode reports the lowest-objective iterate, which need not be the one the
# Jacobian was formed at; firing elsewhere would attach the verdict to an untested point.
at_best_iterate(_, ::AbstractNonlinearTerminationMode, _, _, _) = true
function at_best_iterate(
        tc_cache, mode::AbstractSafeBestNonlinearTerminationMode, fu, u, reltol
    )
    return mode_objective(mode, fu, u, reltol) ≤ tc_cache.best_objective_value
end

function check_gradient_and_update!(cache, J, fu, u)
    tc_cache = cache.termination_cache
    gtol = gradient_tolerance(tc_cache.mode)
    gtol === nothing && return false
    gradient_measure_supported(J) || return false
    at_best_iterate(tc_cache, tc_cache.mode, fu, u, tc_cache.reltol) || return false
    measure = gradient_stationarity_measure(J, fu)
    # Written as `≤` rather than `> ... && return`: a non-finite Jacobian makes the measure
    # `NaN`, and only this direction refuses to terminate on it.
    (measure === nothing || !(measure ≤ gtol)) && return false
    tc_cache.retcode = gradient_stationarity_retcode(tc_cache.leastsq)
    cache.retcode = tc_cache.retcode
    # The retained best is only replaced on a strict improvement, so on a tie it still holds
    # an earlier iterate. Report the one the measure was computed at, or the verdict would
    # describe a point that was never tested.
    update_u!!(tc_cache, u)
    update_from_termination_cache!(tc_cache, cache, tc_cache.mode, u)
    cache.force_stop = true
    return true
end

function check_and_update!(cache, fu, u, uprev)
    return check_and_update!(
        cache.termination_cache, cache, fu, u, uprev, cache.termination_cache.mode
    )
end

function check_and_update!(tc_cache, cache, fu, u, uprev, mode)
    return if tc_cache(fu, u, uprev)
        cache.retcode = tc_cache.retcode
        update_from_termination_cache!(tc_cache, cache, mode, u)
        cache.force_stop = true
    end
end

function update_from_termination_cache!(tc_cache, cache, u = get_u(cache))
    return update_from_termination_cache!(tc_cache, cache, tc_cache.mode, u)
end

function update_from_termination_cache!(
        tc_cache, cache, ::AbstractNonlinearTerminationMode, u = get_u(cache)
    )
    # The residual handed to the termination check is the one at the current iterate, so
    # `cache.fu` already describes `u` and there is nothing to recompute.
    return nothing
end

function update_from_termination_cache!(
        tc_cache, cache, ::AbstractSafeBestNonlinearTerminationMode, u = get_u(cache)
    )
    # `Best` modes retain the lowest-objective iterate, which is usually the one the solve
    # ended on; only a genuine rollback needs the residual recomputed.
    tc_cache.u === nothing && return nothing
    get_u(cache) == tc_cache.u && return nothing
    if SciMLBase.isinplace(cache)
        copyto!(get_u(cache), tc_cache.u)
    else
        SciMLBase.set_u!(cache, tc_cache.u)
    end
    return Utils.evaluate_f!(cache, get_u(cache), cache.p)
end
