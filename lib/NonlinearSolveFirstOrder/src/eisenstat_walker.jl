"""
    EisenstatWalkerForcing2(; η₀ = 0.5, ηₘₐₓ = 0.9, γ = 0.9, α = 2, safeguard = true, safeguard_threshold = 0.1)

Algorithm 2 from the classical work by Eisenstat and Walker (1996) as described by formula (2.6):
    ηₖ = γ * (||rₖ|| / ||rₖ₋₁||)^α

Here the variables denote:
    rₖ residual at iteration k
    η₀   ∈ [0,1) initial value for η
    ηₘₐₓ ∈ [0,1) maximum value for η
    γ    ∈ [0,1) correction factor
    α    ∈ [1,2) correction exponent

Furthermore, the proposed safeguard is implemented:
    ηₖ = max(ηₖ, γ*ηₖ₋₁^α) if γ*ηₖ₋₁^α > safeguard_threshold
to prevent ηₖ from shrinking too fast.
"""
@concrete struct EisenstatWalkerForcing2
    η₀
    ηₘₐₓ
    γ
    α
    safeguard
    safeguard_threshold
end

function EisenstatWalkerForcing2(; η₀ = 0.5, ηₘₐₓ = 0.9, γ = 0.9, α = 2, safeguard = true, safeguard_threshold = 0.1)
    EisenstatWalkerForcing2(η₀, ηₘₐₓ, γ, α, safeguard, safeguard_threshold)
end


@concrete mutable struct EisenstatWalkerForcing2Cache
    p::EisenstatWalkerForcing2
    η
    rnorm
    rnorm_prev
    internalnorm
    verbosity
end



function pre_step_forcing!(cache::EisenstatWalkerForcing2Cache, descend_cache::NonlinearSolveBase.NewtonDescentCache, J, u, fu, iter)
    @SciMLMessage("Eisenstat-Walker forcing residual norm $(cache.rnorm) with rate estimate $(cache.rnorm / cache.rnorm_prev).", cache.verbosity, :forcing)

    # On the first iteration we initialize η with the default initial value and stop.
    if iter == 0
        cache.η = cache.p.η₀
        @SciMLMessage("Eisenstat-Walker initial iteration to η=$(cache.η).", cache.verbosity, :forcing)
        LinearSolve.update_tolerances!(descend_cache.lincache; reltol=cache.η)
        return nothing
    end

    # Store previous
    ηprev = cache.η

    # Formula (2.6)
    # ||r|| > 0 should be guaranteed by the convergence criterion
    (; rnorm, rnorm_prev) = cache
    (; α, γ) = cache.p
    cache.η = γ * (rnorm / rnorm_prev)^α

    # Safeguard 2 to prevent over-solving
    if cache.p.safeguard
        ηsg = γ*ηprev^α
        if ηsg > cache.p.safeguard_threshold && ηsg > cache.η
            cache.η = ηsg
        end
    end

    # Far away from the root we also need to respect η ∈ [0,1)
    cache.η = clamp(cache.η, 0.0, cache.p.ηₘₐₓ)

    @SciMLMessage("Eisenstat-Walker iter $iter update to η=$(cache.η).", cache.verbosity, :forcing)

    # Communicate new relative tolerance to linear solve
    LinearSolve.update_tolerances!(descend_cache.lincache; reltol=cache.η)

    return nothing
end



function post_step_forcing!(cache::EisenstatWalkerForcing2Cache, J, u, fu, δu, iter)
    # Cache previous residual norm
    cache.rnorm_prev = cache.rnorm
    cache.rnorm = cache.internalnorm(fu)

    # @SciMLMessage("Eisenstat-Walker sanity check: $(cache.internalnorm(fu + J*δu)) ≤ $(cache.η * cache.internalnorm(fu)).", cache.verbosity, :linear_verbosity)
end



function InternalAPI.init(
        prob::AbstractNonlinearProblem, alg::EisenstatWalkerForcing2, f, fu, u, p,
        args...; verbose, internalnorm::F = L2_NORM, kwargs...
) where {F}
    fu_norm = internalnorm(fu)

    return EisenstatWalkerForcing2Cache(
        alg, alg.η₀, fu_norm, fu_norm, internalnorm, verbose
    )
end



function InternalAPI.reinit!(
        cache::EisenstatWalkerForcing2Cache; p = cache.p, kwargs...
)
    cache.p = p
end
