function LevenbergMarquardt(; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, damping_initial::Real = 1.0, α_geodesic::Real = 0.75,
        damping_increase_factor::Real = 2.0, damping_decrease_factor::Real = 3.0,
        finite_diff_step_geodesic::Real = 0.1, b_uphill::Real = 1.0, autodiff = nothing,
        min_damping_D::Real = 1e-8, disable_geodesic = False)
    descent = DampedNewtonDescent(; linsolve, precs, initial_damping = damping_initial,
        damping_fn = LevenbergMarquardtDampingFunction(damping_increase_factor,
            damping_decrease_factor, min_damping_D))
    if disable_geodesic === False
        descent = GeodesicAcceleration(descent, finite_diff_step_geodesic, α_geodesic)
    end
    trustregion = LevenbergMarquardtTrustRegion(b_uphill)
    return GeneralizedFirstOrderAlgorithm(; concrete_jac, name = :LevenbergMarquardt,
        trustregion, descent, jacobian_ad = autodiff)
end

@concrete struct LevenbergMarquardtDampingFunction <: AbstractDampingFunction
    increase_factor
    decrease_factor
    min_damping
end

@concrete mutable struct LevenbergMarquardtDampingCache <: AbstractDampingFunctionCache
    increase_factor
    decrease_factor
    min_damping
    λ_factor
    λ
    DᵀD
    J_diag_cache
    J_damped
end

function requires_normal_form_jacobian(::Union{LevenbergMarquardtDampingFunction,
        LevenbergMarquardtDampingCache})
    return false
end
function requires_normal_form_rhs(::Union{LevenbergMarquardtDampingFunction,
        LevenbergMarquardtDampingCache})
    return false
end

function SciMLBase.init(prob::AbstractNonlinearProblem,
        f::LevenbergMarquardtDampingFunction, initial_damping, J, fu, u, ::Val{NF};
        internalnorm::F = DEFAULT_NORM, kwargs...) where {F, NF}
    T = promote_type(eltype(u), eltype(fu))
    DᵀD = __init_diagonal(u, T(f.min_damping))
    if NF
        J_diag_cache = nothing
    else
        @bb J_diag_cache = similar(u)
    end
    if can_setindex(J)
        J_damped = similar(J, length(u), length(u))
    else
        J_damped = J
    end
    return LevenbergMarquardtDampingCache(T(f.increase_factor), T(f.decrease_factor),
        T(f.min_damping), T(f.increase_factor), T(initial_damping), DᵀD, J_diag_cache,
        J_damped)
end

function SciMLBase.solve!(damping::LevenbergMarquardtDampingCache, J, fu, ::Val{false};
        kwargs...)
    if can_setindex(damping.J_diag_cache)
        sum!(abs2, _vec(damping.J_diag_cache), J')
    else
        damping.J_diag_cache = dropdims(sum(abs2, J'; dims = 1); dims = 1)
    end
    damping.DᵀD = __update_LM_diagonal!!(damping.DᵀD, _vec(damping.J_diag_cache))
    @bb @. damping.J_damped = damping.λ * damping.DᵀD
    return damping.J_damped
end

function SciMLBase.solve!(damping::LevenbergMarquardtDampingCache, JᵀJ, fu, ::Val{true};
        kwargs...)
    damping.DᵀD = __update_LM_diagonal!!(damping.DᵀD, JᵀJ)
    @bb @. damping.J_damped = damping.λ * damping.DᵀD
    return damping.J_damped
end

function callback_into_cache!(topcache, cache::LevenbergMarquardtDampingCache, args...)
    if last_step_accepted(topcache.trustregion_cache)
        cache.λ_factor = 1 / cache.decrease_factor
    end
    cache.λ *= cache.λ_factor
    cache.λ_factor = cache.increase_factor
end

@inline __update_LM_diagonal!!(y::Number, x::Number) = max(y, x)
@inline function __update_LM_diagonal!!(y::Diagonal, x::AbstractVector)
    if can_setindex(y.diag)
        @. y.diag = max(y.diag, x)
        return y
    else
        return Diagonal(max.(y.diag, x))
    end
end
@inline function __update_LM_diagonal!!(y::Diagonal, x::AbstractMatrix)
    if can_setindex(y.diag)
        if fast_scalar_indexing(y.diag)
            @inbounds for i in axes(x, 1)
                y.diag[i] = max(y.diag[i], x[i, i])
            end
            return y
        else
            idxs = diagind(x)
            @.. broadcast=false y.diag=max(y.diag, @view(x[idxs]))
            return y
        end
    else
        idxs = diagind(x)
        return Diagonal(@.. broadcast=false max(y.diag, @view(x[idxs])))
    end
end

@inline __init_diagonal(u::Number, v) = oftype(u, v)
@inline __init_diagonal(u::SArray, v) = Diagonal(ones(typeof(vec(u))) * v)
@inline function __init_diagonal(u, v)
    d = similar(vec(u))
    d .= v
    return Diagonal(d)
end
