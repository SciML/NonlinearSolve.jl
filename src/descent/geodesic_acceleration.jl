"""
    GeodesicAcceleration(; descent, finite_diff_step_geodesic, α)

Uses the `descent` algorithm to compute the velocity and acceleration terms for the
geodesic acceleration method. The velocity and acceleration terms are then combined to
compute the descent direction.

This method in its current form was developed for [`LevenbergMarquardt`](@ref). Performance
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
@concrete struct GeodesicAcceleration <: AbstractDescentAlgorithm
    descent
    finite_diff_step_geodesic
    α
end

function Base.show(io::IO, alg::GeodesicAcceleration)
    print(
        io, "GeodesicAcceleration(descent = $(alg.descent), finite_diff_step_geodesic = ",
        "$(alg.finite_diff_step_geodesic), α = $(alg.α))")
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

function __reinit_internal!(
        cache::GeodesicAccelerationCache, args...; p = cache.p, kwargs...)
    cache.p = p
    cache.last_step_accepted = false
end

@internal_caches GeodesicAccelerationCache :descent_cache

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

function __internal_init(prob::AbstractNonlinearProblem, alg::GeodesicAcceleration, J,
        fu, u; shared::Val{N} = Val(1), pre_inverted::Val{INV} = False,
        linsolve_kwargs = (;), abstol = nothing, reltol = nothing,
        internalnorm::F = L2_NORM, kwargs...) where {INV, N, F}
    T = promote_type(eltype(u), eltype(fu))
    @bb δu = similar(u)
    δus = N ≤ 1 ? nothing : map(2:N) do i
        @bb δu_ = similar(u)
    end
    descent_cache = __internal_init(prob, alg.descent, J, fu, u; shared = Val(N * 2),
        pre_inverted, linsolve_kwargs, abstol, reltol, kwargs...)
    @bb Jv = similar(fu)
    @bb fu_cache = copy(fu)
    @bb u_cache = similar(u)
    return GeodesicAccelerationCache(
        δu, δus, descent_cache, prob.f, prob.p, T(alg.α), internalnorm,
        T(alg.finite_diff_step_geodesic), Jv, fu_cache, u_cache, false)
end

function __internal_solve!(
        cache::GeodesicAccelerationCache, J, fu, u, idx::Val{N} = Val(1);
        skip_solve::Bool = false, kwargs...) where {N}
    a, v, δu = get_acceleration(cache, idx), get_velocity(cache, idx), get_du(cache, idx)
    skip_solve && return DescentResult(; δu, extras = (; a, v))
    v = __internal_solve!(
        cache.descent_cache, J, fu, u, Val(2N - 1); skip_solve, kwargs...).δu

    @bb @. cache.u_cache = u + cache.h * v
    cache.fu_cache = evaluate_f!!(cache.f, cache.fu_cache, cache.u_cache, cache.p)

    J !== nothing && @bb(cache.Jv=J × vec(v))
    Jv = _restructure(cache.fu_cache, cache.Jv)
    @bb @. cache.fu_cache = (2 / cache.h) * ((cache.fu_cache - fu) / cache.h - Jv)

    a = __internal_solve!(cache.descent_cache, J, cache.fu_cache, u, Val(2N);
        skip_solve, kwargs..., reuse_A_if_factorization = true).δu

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
