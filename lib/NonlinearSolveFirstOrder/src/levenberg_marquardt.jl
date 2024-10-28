"""
    LevenbergMarquardt(;
        linsolve = nothing, precs = nothing,
        damping_initial::Real = 1.0, α_geodesic::Real = 0.75, disable_geodesic = Val(false),
        damping_increase_factor::Real = 2.0, damping_decrease_factor::Real = 3.0,
        finite_diff_step_geodesic = 0.1, b_uphill::Real = 1.0, min_damping_D::Real = 1e-8,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing
    )

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
        linsolve = nothing, precs = nothing,
        damping_initial::Real = 1.0, α_geodesic::Real = 0.75, disable_geodesic = Val(false),
        damping_increase_factor::Real = 2.0, damping_decrease_factor::Real = 3.0,
        finite_diff_step_geodesic = 0.1, b_uphill::Real = 1.0, min_damping_D::Real = 1e-8,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing
)
    descent = DampedNewtonDescent(;
        linsolve,
        precs,
        initial_damping = damping_initial,
        damping_fn = LevenbergMarquardtDampingFunction(
            damping_increase_factor, damping_decrease_factor, min_damping_D
        )
    )
    if disable_geodesic isa Val{false}
        descent = GeodesicAcceleration(descent, finite_diff_step_geodesic, α_geodesic)
    end
    trustregion = LevenbergMarquardtTrustRegion(b_uphill)
    return GeneralizedFirstOrderAlgorithm(;
        trustregion,
        descent,
        autodiff,
        vjp_autodiff,
        jvp_autodiff,
        name = :LevenbergMarquardt
    )
end

@concrete struct LevenbergMarquardtDampingFunction <: AbstractDampingFunction
    increase_factor
    decrease_factor
    min_damping
end

function InternalAPI.init(
        prob::AbstractNonlinearProblem, f::LevenbergMarquardtDampingFunction,
        initial_damping, J, fu, u, normal_form::Val; kwargs...
)
    T = promote_type(eltype(u), eltype(fu))
    DᵀD = init_levenberg_marquardt_diagonal(u, T(f.min_damping))
    if normal_form isa Val{true}
        J_diag_cache = nothing
    else
        @bb J_diag_cache = similar(u)
    end
    J_damped = T(initial_damping) .* DᵀD
    return LevenbergMarquardtDampingCache(
        T(f.increase_factor), T(f.decrease_factor), T(f.min_damping),
        T(f.increase_factor), T(initial_damping), DᵀD, J_diag_cache, J_damped, f,
        T(initial_damping)
    )
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

function InternalAPI.reinit!(cache::LevenbergMarquardtDampingCache, args...; kwargs...)
    cache.λ = cache.initial_damping
    cache.λ_factor = cache.damping_f.increase_factor
    if !(cache.DᵀD isa Number)
        if ArrayInterface.can_setindex(cache.DᵀD.diag)
            cache.DᵀD.diag .= cache.min_damping
        else
            cache.DᵀD = Diagonal(ones(typeof(cache.DᵀD.diag)) * cache.min_damping)
        end
    end
    cache.J_damped = cache.λ .* cache.DᵀD
    return
end

function NonlinearSolveBase.requires_normal_form_jacobian(::Union{
        LevenbergMarquardtDampingFunction, LevenbergMarquardtDampingCache})
    return false
end
function NonlinearSolveBase.requires_normal_form_rhs(::Union{
        LevenbergMarquardtDampingFunction, LevenbergMarquardtDampingCache})
    return false
end
function NonlinearSolveBase.returns_norm_form_damping(::Union{
        LevenbergMarquardtDampingFunction, LevenbergMarquardtDampingCache})
    return true
end

(damping::LevenbergMarquardtDampingCache)(::Nothing) = damping.J_damped

function InternalAPI.solve!(
        cache::LevenbergMarquardtDampingCache, J, fu, ::Val{false}; kwargs...
)
    if ArrayInterface.can_setindex(cache.J_diag_cache)
        sum!(abs2, Utils.safe_vec(cache.J_diag_cache), J')
    elseif cache.J_diag_cache isa Number
        cache.J_diag_cache = abs2(J)
    else
        cache.J_diag_cache = dropdims(sum(abs2, J'; dims = 1); dims = 1)
    end
    cache.DᵀD = update_levenberg_marquardt_diagonal!!(
        cache.DᵀD, Utils.safe_vec(cache.J_diag_cache)
    )
    @bb @. cache.J_damped = cache.λ * cache.DᵀD
    return cache.J_damped
end

function InternalAPI.solve!(
        cache::LevenbergMarquardtDampingCache, JᵀJ, fu, ::Val{true}; kwargs...
)
    cache.DᵀD = update_levenberg_marquardt_diagonal!!(cache.DᵀD, JᵀJ)
    @bb @. cache.J_damped = cache.λ * cache.DᵀD
    return cache.J_damped
end

function NonlinearSolveBase.callback_into_cache!(
        topcache, cache::LevenbergMarquardtDampingCache, args...
)
    if NonlinearSolveBase.last_step_accepted(topcache.trustregion_cache) &&
       NonlinearSolveBase.last_step_accepted(topcache.descent_cache)
        cache.λ_factor = 1 / cache.decrease_factor
    end
    cache.λ *= cache.λ_factor
    cache.λ_factor = cache.increase_factor
end

"""
    LevenbergMarquardtTrustRegion(b_uphill)

Trust Region method for [`LevenbergMarquardt`](@ref). This method is tightly coupled with
the Levenberg-Marquardt method and works by directly updating the damping parameter instead
of specifying a trust region radius.

### Arguments

  - `b_uphill`: a factor that determines if a step is accepted or rejected. The standard
    choice in the Levenberg-Marquardt method is to accept all steps that decrease the cost
    and reject all steps that increase the cost. Although this is a natural and safe choice,
    it is often not the most efficient. Therefore downhill moves are always accepted, but
    uphill moves are only conditionally accepted. To decide whether an uphill move will be
    accepted at each iteration ``i``, we compute
    ``\\beta_i = \\cos(v_{\\text{new}}, v_{\\text{old}})``, which denotes the cosine angle
    between the proposed velocity ``v_{\\text{new}}`` and the velocity of the last accepted
    step ``v_{\\text{old}}``. The idea is to accept uphill moves if the angle is small. To
    specify, uphill moves are accepted if
    ``(1-\\beta_i)^{b_{\\text{uphill}}} C_{i+1} \\le C_i``, where ``C_i`` is the cost at
    iteration ``i``. Reasonable choices for `b_uphill` are `1.0` or `2.0`, with
    `b_uphill = 2.0` allowing higher uphill moves than `b_uphill = 1.0`. When
    `b_uphill = 0.0`, no uphill moves will be accepted. Defaults to `1.0`. See Section 4 of
    [transtrum2012improvements](@citet).
"""
@concrete struct LevenbergMarquardtTrustRegion <: AbstractTrustRegionMethod
    β_uphill
end

function InternalAPI.init(
        prob::AbstractNonlinearProblem, alg::LevenbergMarquardtTrustRegion,
        f::NonlinearFunction, fu, u, p, args...;
        stats, internalnorm::F = L2_NORM, kwargs...
) where {F}
    T = promote_type(eltype(u), eltype(fu))
    @bb v = copy(u)
    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    return LevenbergMarquardtTrustRegionCache(
        f, p, T(Inf), v, T(Inf), internalnorm, T(alg.β_uphill), false,
        u_cache, fu_cache, stats
    )
end

@concrete mutable struct LevenbergMarquardtTrustRegionCache <:
                         AbstractTrustRegionMethodCache
    f
    p
    loss_old
    v_cache
    norm_v_old
    internalnorm
    β_uphill
    last_step_accepted::Bool
    u_cache
    fu_cache
    stats::NLStats
end

function InternalAPI.reinit!(
        cache::LevenbergMarquardtTrustRegionCache; p = cache.p, u0 = cache.v_cache, kwargs...
)
    cache.p = p
    @bb copyto!(cache.v_cache, u0)
    cache.loss_old = oftype(cache.loss_old, Inf)
    cache.norm_v_old = oftype(cache.norm_v_old, Inf)
    cache.last_step_accepted = false
end

function InternalAPI.solve!(
        cache::LevenbergMarquardtTrustRegionCache, J, fu, u, δu, descent_stats
)
    # This should be true if Geodesic Acceleration is being used
    v = hasfield(typeof(descent_stats), :v) ? descent_stats.v : δu
    norm_v = cache.internalnorm(v)
    β = dot(v, cache.v_cache) / (norm_v * cache.norm_v_old)

    @bb @. cache.u_cache = u + δu
    cache.fu_cache = Utils.evaluate_f!!(cache.f, cache.fu_cache, cache.u_cache, cache.p)
    cache.stats.nf += 1

    loss = cache.internalnorm(cache.fu_cache)

    if (1 - β)^cache.β_uphill * loss ≤ cache.loss_old  # Accept Step
        cache.last_step_accepted = true
        cache.norm_v_old = norm_v
        @bb copyto!(cache.v_cache, v)
    else
        cache.last_step_accepted = false
    end

    return cache.last_step_accepted, cache.u_cache, cache.fu_cache
end

update_levenberg_marquardt_diagonal!!(y::Number, x::Number) = max(y, x)
function update_levenberg_marquardt_diagonal!!(y::Diagonal, x::AbstractVecOrMat)
    if ArrayInterface.can_setindex(y.diag)
        if ArrayInterface.fast_scalar_indexing(y.diag)
            if ndims(x) == 1
                @simd ivdep for i in axes(x, 1)
                    @inbounds y.diag[i] = max(y.diag[i], x[i])
                end
            else
                @simd ivdep for i in axes(x, 1)
                    @inbounds y.diag[i] = max(y.diag[i], x[i, i])
                end
            end
        else
            if ndims(x) == 1
                @. y.diag = max(y.diag, x)
            else
                y.diag .= max.(y.diag, @view(x[diagind(x)]))
            end
        end
        return y
    end
    ndims(x) == 1 && return Diagonal(max.(y.diag, x))
    return Diagonal(max.(y.diag, @view(x[diagind(x)])))
end

init_levenberg_marquardt_diagonal(u::Number, v) = oftype(u, v)
init_levenberg_marquardt_diagonal(u::SArray, v) = Diagonal(ones(typeof(vec(u))) * v)
function init_levenberg_marquardt_diagonal(u, v)
    d = similar(vec(u))
    d .= v
    return Diagonal(d)
end
