const RelNormModes = Union{
    RelNormTerminationMode, RelNormSafeTerminationMode, RelNormSafeBestTerminationMode}
const AbsNormModes = Union{
    AbsNormTerminationMode, AbsNormSafeTerminationMode, AbsNormSafeBestTerminationMode}

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
    u_diff_cache::uType
end

function update_u!!(cache::NonlinearTerminationModeCache, u)
    cache.u === nothing && return
    if cache.u isa AbstractArray && ArrayInterface.can_setindex(cache.u)
        copyto!(cache.u, u)
    else
        cache.u .= u
    end
end

function SciMLBase.init(du::Union{AbstractArray{T}, T}, u::Union{AbstractArray{T}, T},
        mode::AbstractNonlinearTerminationMode, saved_value_prototype...;
        abstol = nothing, reltol = nothing, kwargs...) where {T <: Number}
    error("Not yet implemented...")
end

function SciMLBase.reinit!(
        cache::NonlinearTerminationModeCache, du, u, saved_value_prototype...;
        abstol = nothing, reltol = nothing, kwargs...)
    error("Not yet implemented...")
end

## This dispatch is needed based on how Terminating Callback works!
function (cache::NonlinearTerminationModeCache)(
        integrator::AbstractODEIntegrator, abstol::Number, reltol::Number, min_t)
    if min_t === nothing || integrator.t ≥ min_t
        return cache(cache.mode, SciMLBase.get_du(integrator),
            integrator.u, integrator.uprev, abstol, reltol)
    end
    return false
end
function (cache::NonlinearTerminationModeCache)(du, u, uprev, args...)
    return cache(cache.mode, du, u, uprev, cache.abstol, cache.reltol, args...)
end

function (cache::NonlinearTerminationModeCache)(
        mode::AbstractNonlinearTerminationMode, du, u, uprev, abstol, reltol, args...)
    if check_convergence(mode, du, u, uprev, abstol, reltol)
        cache.retcode = ReturnCode.Success
        return true
    end
    return false
end

function (cache::NonlinearTerminationModeCache)(
        mode::AbstractSafeNonlinearTerminationMode, du, u, uprev, abstol, reltol, args...)
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
    cache.nsteps == 1 && (cache.initial_objective = objective)
    cache.objectives_trace[mod1(cache.nsteps, length(cache.objectives_trace))] = objective

    if objective ≤ mode.patience_objective_multiplier * criteria &&
       cache.nsteps > mode.patience_steps
        if cache.nsteps < length(cache.objectives_trace)
            min_obj, max_obj = extrema(@view(cache.objectives_trace[1:(cache.nsteps)]))
        else
            min_obj, max_obj = extrema(cache.objectives_trace)
        end
        if min_obj < mode.min_max_factor * max_obj
            cache.retcode = ReturnCode.Stalled
            return true
        end
    end

    # Test for stalling if that is enabled
    if cache.step_norm_trace !== nothing
        if ArrayInterface.can_setindex(cache.u_diff_cache) && !(u isa Number)
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
                cache.retcode = ReturnCode.Stalled
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
        return all(@closure(xy->begin
            x, y = xy
            return abs(y) ≤ reltol * abs(x + y)
        end), zip(uₙ, duₙ))
    else # using mapreduce here will almost certainly be faster on GPUs
        return mapreduce(
            @closure((xᵢ, yᵢ)->(abs(yᵢ) ≤ reltol * abs(xᵢ + yᵢ))), *, uₙ, duₙ; init = true)
    end
end
function check_convergence(::AbsTerminationMode, duₙ, _, __, abstol, ___)
    return all(@closure(x->abs(x) ≤ abstol), duₙ)
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
