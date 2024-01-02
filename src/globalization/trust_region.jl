@concrete struct LevenbergMarquardtTrustRegion <: AbstractTrustRegionMethod
    β_uphill
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
end

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::LevenbergMarquardtTrustRegion,
        f::F, fu, u, p, args...; internalnorm::IF = DEFAULT_NORM, kwargs...) where {F, IF}
    T = promote_type(eltype(u), eltype(fu))
    @bb v = similar(u)
    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    return LevenbergMarquardtTrustRegionCache(f, p, T(Inf), v, T(Inf), internalnorm,
        alg.β_uphill, false, u_cache, fu_cache)
end

function SciMLBase.solve!(cache::LevenbergMarquardtTrustRegionCache, u, δu, damping_stats)
    # This should be true if Geodesic Acceleration is being used
    v = hasfield(typeof(damping_stats), :v) ? damping_stats.v : δu
    norm_v = cache.internalnorm(v)
    β = dot(v, cache.v_cache) / (norm_v * cache.norm_v_old)

    @bb @. cache.u_cache = u + δu
    cache.fu_cache = evaluate_f!!(cache.f, cache.fu_cache, cache.u_cache, cache.p)

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
