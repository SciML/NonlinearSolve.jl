"""
    NoChangeInStateReset(;
        nsteps::Int = 3, reset_tolerance = nothing,
        check_du::Bool = true, check_dfu::Bool = true
    )

Recommends a reset if the state or the function value has not changed significantly in
`nsteps` steps. This is used in [`Broyden`](@ref).

### Keyword Arguments

  - `nsteps`: the number of steps to check for no change. Defaults to `3`.
  - `reset_tolerance`: the tolerance for the reset check. Defaults to
    `eps(real(eltype(u)))^(3 // 4)`.
  - `check_du`: whether to check the state. Defaults to `true`.
  - `check_dfu`: whether to check the function value. Defaults to `true`.
"""
@kwdef @concrete struct NoChangeInStateReset <: AbstractResetCondition
    nsteps::Int = 3
    reset_tolerance = nothing
    check_du::Bool = true
    check_dfu::Bool = true
end

function InternalAPI.init(
        condition::NoChangeInStateReset, J, fu, u, du, args...; kwargs...
)
    if condition.check_dfu
        @bb dfu = copy(fu)
    else
        dfu = fu
    end
    T = real(eltype(u))
    tol = condition.reset_tolerance === nothing ? eps(T)^(3 // 4) :
          T(condition.reset_tolerance)
    return NoChangeInStateResetCache(dfu, tol, condition, 0, 0)
end

@concrete mutable struct NoChangeInStateResetCache <: AbstractResetConditionCache
    dfu
    reset_tolerance
    condition <: NoChangeInStateReset
    steps_since_change_du::Int
    steps_since_change_dfu::Int
end

function InternalAPI.reinit!(cache::NoChangeInStateResetCache; u0 = nothing, kwargs...)
    if u0 !== nothing && cache.condition.reset_tolerance === nothing
        cache.reset_tolerance = eps(real(eltype(u0)))^(3 // 4)
    end
    cache.steps_since_change_dfu = 0
    cache.steps_since_change_du = 0
end

function InternalAPI.solve!(cache::NoChangeInStateResetCache, J, fu, u, du; kwargs...)
    cond = ≤(cache.reset_tolerance) ∘ abs
    if cache.condition.check_du
        if any(cond, du)
            cache.steps_since_change_du += 1
            if cache.steps_since_change_du ≥ cache.condition.nsteps
                cache.steps_since_change_du = 0
                cache.steps_since_change_dfu = 0
                return true
            end
        else
            cache.steps_since_change_du = 0
            cache.steps_since_change_dfu = 0
        end
    end
    if cache.condition.check_dfu
        @bb @. cache.dfu = fu - cache.dfu
        if any(cond, cache.dfu)
            cache.steps_since_change_dfu += 1
            if cache.steps_since_change_dfu ≥ cache.condition.nsteps
                cache.steps_since_change_dfu = 0
                cache.steps_since_change_du = 0
                @bb copyto!(cache.dfu, fu)
                return true
            end
        else
            cache.steps_since_change_dfu = 0
            cache.steps_since_change_du = 0
        end
        @bb copyto!(cache.dfu, fu)
    end
    return false
end

"""
    IllConditionedJacobianReset()

Recommend resetting the Jacobian if the current jacobian is ill-conditioned. This is used
in [`Klement`](@ref).
"""
struct IllConditionedJacobianReset <: AbstractResetCondition end

function InternalAPI.init(
        condition::IllConditionedJacobianReset, J, fu, u, du, args...; kwargs...
)
    condition_number_threshold = J isa AbstractMatrix ? inv(eps(real(eltype(J))))^(1 // 2) :
                                 nothing
    return IllConditionedJacobianResetCache(condition_number_threshold)
end

@concrete struct IllConditionedJacobianResetCache <: AbstractResetConditionCache
    condition_number_threshold
end

# NOTE: we don't need a reinit! since we establish the threshold based on the eltype

function InternalAPI.solve!(
        cache::IllConditionedJacobianResetCache, J, fu, u, du; kwargs...
)
    J isa Number && return iszero(J)
    J isa Diagonal && return any(iszero, diag(J))
    J isa AbstractVector && return any(iszero, J)
    J isa AbstractMatrix &&
        return Utils.condition_number(J) ≥ cache.condition_number_threshold
    return false
end
