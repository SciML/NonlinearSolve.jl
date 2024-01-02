"""
    GeodesicAcceleration(; descent, finite_diff_step_geodesic, α)

Uses the `descent` algorithm to compute the velocity and acceleration terms for the
geodesic acceleration method. The velocity and acceleration terms are then combined to
compute the descent direction.

This method in its current form was developed for [`LevenbergMarquardt`](@ref). Performance
for other methods are not theorectically or experimentally verified.

### References

[1] Transtrum, Mark K., and James P. Sethna. "Improvements to the Levenberg-Marquardt
algorithm for nonlinear least-squares minimization." arXiv preprint arXiv:1201.5885
(2012).
"""
@concrete struct GeodesicAcceleration <: AbstractDescentAlgorithm
    descent
    finite_diff_step_geodesic
    α
end

supports_trust_region(::GeodesicAcceleration) = true

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
end

function callback_into_cache!(cache, internalcache::GeodesicAccelerationCache, args...)
    callback_into_cache!(cache, internalcache.descent_cache, internalcache, args...)
end

get_velocity(cache::GeodesicAccelerationCache) = get_du(cache.descent_cache, Val(1))
function set_velocity!(cache::GeodesicAccelerationCache, δv)
    set_du!(cache.descent_cache, δv, Val(1))
end
function get_velocity(cache::GeodesicAccelerationCache, ::Val{N}) where {N}
    get_du(cache.descent_cache, Val(2N - 1))
end
function set_velocity!(cache::GeodesicAccelerationCache, δv, ::Val{N}) where {N}
    set_du!(cache.descent_cache, δv, Val(2N - 1))
end
get_acceleration(cache::GeodesicAccelerationCache) = get_du(cache.descent_cache, Val(2))
function set_acceleration!(cache::GeodesicAccelerationCache, δa)
    set_du!(cache.descent_cache, δa, Val(2))
end
function get_acceleration(cache::GeodesicAccelerationCache, ::Val{N}) where {N}
    get_du(cache.descent_cache, Val(2N))
end
function set_acceleration!(cache::GeodesicAccelerationCache, δa, ::Val{N}) where {N}
    set_du!(cache.descent_cache, δa, Val(2N))
end

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::GeodesicAcceleration, J, fu, u;
        shared::Val{N} = Val(1), pre_inverted::Val{INV} = False, linsolve_kwargs = (;),
        abstol = nothing, reltol = nothing, internalnorm::F = DEFAULT_NORM,
        kwargs...) where {INV, N, F}
    T = promote_type(eltype(u), eltype(fu))
    @bb δu = similar(u)
    δus = N ≤ 1 ? nothing : map(2:N) do i
        @bb δu_ = similar(u)
    end
    descent_cache = init(prob, alg.descent, J, fu, u; shared = Val(N * 2), pre_inverted,
        linsolve_kwargs, abstol, reltol, kwargs...)
    @bb Jv = similar(fu)
    @bb fu_cache = copy(fu)
    @bb u_cache = similar(u)
    return GeodesicAccelerationCache(δu, δus, descent_cache, prob.f, prob.p, T(alg.α),
        internalnorm, T(alg.finite_diff_step_geodesic), Jv, fu_cache, u_cache)
end

function SciMLBase.solve!(cache::GeodesicAccelerationCache, J, fu, u,
        idx::Val{N} = Val(1); skip_solve::Bool = false, kwargs...) where {N}
    a, v, δu = get_acceleration(cache, idx), get_velocity(cache, idx), get_du(cache, idx)
    skip_solve && return δu, true, (; a, v)
    v, _, _ = solve!(cache.descent_cache, J, fu, Val(2N - 1); skip_solve, kwargs...)

    @bb @. cache.u_cache = u + cache.h * v
    cache.fu_cache = evaluate_f!!(cache.f, cache.fu_cache, cache.u_cache, cache.p)

    J !== nothing && @bb(cache.Jv=J × vec(v))
    Jv = _restructure(cache.fu_cache, cache.Jv)
    @bb @. cache.fu_cache = (2 / cache.h) * ((cache.fu_cache - fu) / cache.h - Jv)
    # FIXME: Deepcopy, J
    a, _, _ = solve!(deepcopy(cache.descent_cache), J, cache.fu_cache, Val(2N); skip_solve,
        kwargs...)

    norm_v = cache.internalnorm(v)
    norm_a = cache.internalnorm(a)

    if 2 * norm_a ≤ norm_v * cache.α
        @bb @. δu = v + a / 2
        set_du!(cache, δu, idx)
        return δu, true, (; a, v)
    end
    return δu, false, (; a, v)
end
