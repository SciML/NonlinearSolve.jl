"""
    GeodesicAcceleration(; descent, finite_diff_step_geodesic, α)

Uses the `descent` algorithm to compute the velocity and acceleration terms for the
geodesic acceleration method. The velocity and acceleration terms are then combined to
compute the descent direction.

This method in its current form was developed for `LevenbergMarquardt`. Performance
for other methods are not theorectically or experimentally verified.

### Keyword Arguments

  - `descent`: the descent algorithm to use for computing the velocity and acceleration.
  - `finite_diff_step_geodesic`: the step size used for finite differencing used to
    calculate the geodesic acceleration. Defaults to `0.1` which means that the step size is
    approximately 10% of the first-order step. See Section 3 of [1].
  - `α`: a factor that determines if a step is accepted or rejected. To incorporate
    geodesic acceleration as an addition to the Levenberg-Marquardt algorithm, it is
    necessary that acceptable steps meet the condition
    ``\\frac{2||a||}{||v||} \\le \\alpha_{\\text{geodesic}}``, where ``a`` is the geodesic
    acceleration, ``v`` is the Levenberg-Marquardt algorithm's step (velocity along a
    geodesic path) and `α_geodesic` is some number of order `1`. For most problems
    `α_geodesic = 0.75` is a good value but for problems where convergence is difficult
    `α_geodesic = 0.1` is an effective choice. Defaults to `0.75`. See Section 3 of
    [transtrum2012improvements](@citet).
"""
@concrete struct GeodesicAcceleration <: AbstractDescentDirection
    descent
    finite_diff_step_geodesic
    α
end

supports_trust_region(::GeodesicAcceleration) = true

get_linear_solver(alg::GeodesicAcceleration) = get_linear_solver(alg.descent)

@concrete mutable struct GeodesicAccelerationCache <: AbstractDescentCache
    δu
    δus
    descent_cache
    f
    p
    α
    internalnorm
    h
    Jv
    fu_cache
    u_cache
    last_step_accepted::Bool
end

function InternalAPI.reinit_self!(cache::GeodesicAccelerationCache; p = cache.p, kwargs...)
    cache.p = p
    cache.last_step_accepted = false
end

@internal_caches GeodesicAccelerationCache :descent_cache

function get_velocity(cache::GeodesicAccelerationCache)
    return SciMLBase.get_du(cache.descent_cache, Val(1))
end
function get_velocity(cache::GeodesicAccelerationCache, ::Val{N}) where {N}
    return SciMLBase.get_du(cache.descent_cache, Val(2N - 1))
end
function get_acceleration(cache::GeodesicAccelerationCache)
    return SciMLBase.get_du(cache.descent_cache, Val(2))
end
function get_acceleration(cache::GeodesicAccelerationCache, ::Val{N}) where {N}
    return SciMLBase.get_du(cache.descent_cache, Val(2N))
end

function InternalAPI.init(
        prob::AbstractNonlinearProblem, alg::GeodesicAcceleration, J, fu, u;
        shared::Val = Val(1), pre_inverted::Val = Val(false), linsolve_kwargs = (;),
        abstol = nothing, reltol = nothing,
        internalnorm::F = L2_NORM, kwargs...
) where {F}
    T = promote_type(eltype(u), eltype(fu))
    @bb δu = similar(u)
    δus = Utils.unwrap_val(shared) ≤ 1 ? nothing : map(2:Utils.unwrap_val(shared)) do i
        @bb δu_ = similar(u)
    end
    descent_cache = InternalAPI.init(
        prob, alg.descent, J, fu, u;
        shared = Val(2 * Utils.unwrap_val(shared)), pre_inverted, linsolve_kwargs,
        abstol, reltol,
        kwargs...
    )
    @bb Jv = similar(fu)
    @bb fu_cache = copy(fu)
    @bb u_cache = similar(u)
    return GeodesicAccelerationCache(
        δu, δus, descent_cache, prob.f, prob.p, T(alg.α), internalnorm,
        T(alg.finite_diff_step_geodesic), Jv, fu_cache, u_cache, false
    )
end

function InternalAPI.solve!(
        cache::GeodesicAccelerationCache, J, fu, u, idx::Val = Val(1);
        skip_solve::Bool = false, kwargs...
)
    a = get_acceleration(cache, idx)
    v = get_velocity(cache, idx)
    δu = SciMLBase.get_du(cache, idx)
    skip_solve && return DescentResult(; δu, extras = (; a, v))

    v = InternalAPI.solve!(
        cache.descent_cache, J, fu, u, Val(2 * Utils.unwrap_val(idx) - 1);
        skip_solve, kwargs...
    ).δu

    @bb @. cache.u_cache = u + cache.h * v
    cache.fu_cache = Utils.evaluate_f!!(cache.f, cache.fu_cache, cache.u_cache, cache.p)

    J !== nothing && @bb(cache.Jv=J × vec(v))
    Jv = Utils.restructure(cache.fu_cache, cache.Jv)
    @bb @. cache.fu_cache = (2 / cache.h) * ((cache.fu_cache - fu) / cache.h - Jv)

    a = InternalAPI.solve!(
        cache.descent_cache, J, cache.fu_cache, u, Val(2 * Utils.unwrap_val(idx));
        skip_solve, kwargs..., reuse_A_if_factorization = true
    ).δu

    norm_v = cache.internalnorm(v)
    norm_a = cache.internalnorm(a)

    if 2 * norm_a ≤ norm_v * cache.α
        @bb @. δu = v + a / 2
        set_du!(cache, δu, idx)
        cache.last_step_accepted = true
    else
        cache.last_step_accepted = false
    end

    return DescentResult(; δu, success = cache.last_step_accepted, extras = (; a, v))
end
