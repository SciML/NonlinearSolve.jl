"""
    LevenbergMarquardt(; linsolve = nothing,
        precs = DEFAULT_PRECS, damping_initial::Real = 1.0, α_geodesic::Real = 0.75,
        damping_increase_factor::Real = 2.0, damping_decrease_factor::Real = 3.0,
        finite_diff_step_geodesic = 0.1, b_uphill::Real = 1.0, autodiff = nothing,
        min_damping_D::Real = 1e-8, disable_geodesic = Val(false))

An advanced Levenberg-Marquardt implementation with the improvements suggested in
[transtrum2012improvements](@citet). Designed for large-scale and numerically-difficult
nonlinear systems.

### Keyword Arguments

  - `damping_initial`: the starting value for the damping factor. The damping factor is
    inversely proportional to the step size. The damping factor is adjusted during each
    iteration. Defaults to `1.0`. See Section 2.1 of [transtrum2012improvements](@citet).
  - `damping_increase_factor`: the factor by which the damping is increased if a step is
    rejected. Defaults to `2.0`. See Section 2.1 of [transtrum2012improvements](@citet).
  - `damping_decrease_factor`: the factor by which the damping is decreased if a step is
    accepted. Defaults to `3.0`. See Section 2.1 of [transtrum2012improvements](@citet).
  - `min_damping_D`: the minimum value of the damping terms in the diagonal damping matrix
    `DᵀD`, where `DᵀD` is given by the largest diagonal entries of `JᵀJ` yet encountered,
    where `J` is the Jacobian. It is suggested by [transtrum2012improvements](@citet) to use
    a minimum value of the elements in `DᵀD` to prevent the damping from being too small.
    Defaults to `1e-8`.
  - `disable_geodesic`: Disables Geodesic Acceleration if set to `Val(true)`. It provides
    a way to trade-off robustness for speed, though in most situations Geodesic Acceleration
    should not be disabled.

For the remaining arguments, see [`GeodesicAcceleration`](@ref) and
[`NonlinearSolve.LevenbergMarquardtTrustRegion`](@ref) documentations.
"""
function LevenbergMarquardt(;
        linsolve = nothing, precs = DEFAULT_PRECS, damping_initial::Real = 1.0,
        α_geodesic::Real = 0.75, damping_increase_factor::Real = 2.0,
        damping_decrease_factor::Real = 3.0, finite_diff_step_geodesic = 0.1,
        b_uphill::Real = 1.0, autodiff = nothing,
        min_damping_D::Real = 1e-8, disable_geodesic = False)
    descent = DampedNewtonDescent(; linsolve,
        precs,
        initial_damping = damping_initial,
        damping_fn = LevenbergMarquardtDampingFunction(
            damping_increase_factor, damping_decrease_factor, min_damping_D))
    if disable_geodesic === False
        descent = GeodesicAcceleration(descent, finite_diff_step_geodesic, α_geodesic)
    end
    trustregion = LevenbergMarquardtTrustRegion(b_uphill)
    return GeneralizedFirstOrderAlgorithm(;
        concrete_jac = true, name = :LevenbergMarquardt,
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
    damping_f
    initial_damping
end

function reinit_cache!(cache::LevenbergMarquardtDampingCache, args...; kwargs...)
    cache.λ = cache.initial_damping
    cache.λ_factor = cache.damping_f.increase_factor
    if !(cache.DᵀD isa Number)
        if can_setindex(cache.DᵀD.diag)
            cache.DᵀD.diag .= cache.min_damping
        else
            cache.DᵀD = Diagonal(ones(typeof(cache.DᵀD.diag)) * cache.min_damping)
        end
    end
    cache.J_damped = cache.λ .* cache.DᵀD
end

function requires_normal_form_jacobian(::Union{
        LevenbergMarquardtDampingFunction, LevenbergMarquardtDampingCache})
    return false
end
function requires_normal_form_rhs(::Union{
        LevenbergMarquardtDampingFunction, LevenbergMarquardtDampingCache})
    return false
end
function returns_norm_form_damping(::Union{
        LevenbergMarquardtDampingFunction, LevenbergMarquardtDampingCache})
    return true
end

function __internal_init(
        prob::AbstractNonlinearProblem, f::LevenbergMarquardtDampingFunction,
        initial_damping, J, fu, u, ::Val{NF};
        internalnorm::F = L2_NORM, kwargs...) where {F, NF}
    T = promote_type(eltype(u), eltype(fu))
    DᵀD = __init_diagonal(u, T(f.min_damping))
    if NF
        J_diag_cache = nothing
    else
        @bb J_diag_cache = similar(u)
    end
    J_damped = T(initial_damping) .* DᵀD
    return LevenbergMarquardtDampingCache(
        T(f.increase_factor), T(f.decrease_factor), T(f.min_damping), T(f.increase_factor),
        T(initial_damping), DᵀD, J_diag_cache, J_damped, f, T(initial_damping))
end

(damping::LevenbergMarquardtDampingCache)(::Nothing) = damping.J_damped

function __internal_solve!(
        damping::LevenbergMarquardtDampingCache, J, fu, ::Val{false}; kwargs...)
    if __can_setindex(damping.J_diag_cache)
        sum!(abs2, _vec(damping.J_diag_cache), J')
    elseif damping.J_diag_cache isa Number
        damping.J_diag_cache = abs2(J)
    else
        damping.J_diag_cache = dropdims(sum(abs2, J'; dims = 1); dims = 1)
    end
    damping.DᵀD = __update_LM_diagonal!!(damping.DᵀD, _vec(damping.J_diag_cache))
    @bb @. damping.J_damped = damping.λ * damping.DᵀD
    return damping.J_damped
end

function __internal_solve!(
        damping::LevenbergMarquardtDampingCache, JᵀJ, fu, ::Val{true}; kwargs...)
    damping.DᵀD = __update_LM_diagonal!!(damping.DᵀD, JᵀJ)
    @bb @. damping.J_damped = damping.λ * damping.DᵀD
    return damping.J_damped
end

function callback_into_cache!(topcache, cache::LevenbergMarquardtDampingCache, args...)
    if last_step_accepted(topcache.trustregion_cache) &&
       last_step_accepted(topcache.descent_cache)
        cache.λ_factor = 1 / cache.decrease_factor
    end
    cache.λ *= cache.λ_factor
    cache.λ_factor = cache.increase_factor
end

@inline __update_LM_diagonal!!(y::Number, x::Number) = max(y, x)
@inline function __update_LM_diagonal!!(y::Diagonal, x::AbstractVector)
    if __can_setindex(y.diag)
        @. y.diag = max(y.diag, x)
        return y
    else
        return Diagonal(max.(y.diag, x))
    end
end
@inline function __update_LM_diagonal!!(y::Diagonal, x::AbstractMatrix)
    if __can_setindex(y.diag)
        if fast_scalar_indexing(y.diag)
            @simd for i in axes(x, 1)
                @inbounds y.diag[i] = max(y.diag[i], x[i, i])
            end
            return y
        else
            y .= max.(y.diag, @view(x[diagind(x)]))
            return y
        end
    else
        return Diagonal(max.(y.diag, @view(x[diagind(x)])))
    end
end

@inline __init_diagonal(u::Number, v) = oftype(u, v)
@inline __init_diagonal(u::SArray, v) = Diagonal(ones(typeof(vec(u))) * v)
@inline function __init_diagonal(u, v)
    d = similar(vec(u))
    d .= v
    return Diagonal(d)
end
