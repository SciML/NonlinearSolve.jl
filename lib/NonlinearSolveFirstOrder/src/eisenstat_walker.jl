"""
    EisenstatWalkerForcing2(; η₀, γ, α)

Algorithm 2 from the classical work by Eisenstat and Walker (1996) as described by formula (2.6).
"""
@concrete struct EisenstatWalkerForcing2
    η₀
    ηₘₐₓ
    γ
    α
    safeguard
    safeguard_threshold
end

function EisenstatWalkerForcing2(; η₀ = 0.5, ηₘₐₓ = 0.99, γ = 0.9, α = 2, safeguard = true, safeguard_threshold = 0.1)
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
    # On the first iteration we initialize η with the default initial value and stop.
    if iter == 0
        cache.η = cache.p.η₀
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
    cache.η = clamp(cache.η, 0.0, cache.ηₘₐₓ)

    @SciMLMessage("Eisenstat-Walker update to η=$(cache.η).", cache.verbosity, :linear_verbosity)

    # Communicate new relative tolerance to linear solve
    LinearSolve.update_tolerances!(descend_cache.lincache; reltol=cache.η)

    return nothing
end



function post_step_forcing!(cache::EisenstatWalkerForcing2Cache, J, u, fu, δu, iter)
    # Cache previous residual norm
    cache.rnorm_prev = cache.rnorm
    # And update the current one
    cache.rnorm = cache.internalnorm(fu)
end



function InternalAPI.init(
        prob::AbstractNonlinearProblem, alg::EisenstatWalkerForcing2, f, fu, u, p,
        args...; verbose, internalnorm::F = L2_NORM, kwargs...
) where {F}
    fu_norm = internalnorm(fu)

    return EisenstatWalkerForcing2Cache(
        alg, alg.η₀, fu_norm, fu_norm, internalnorm
    )
end



function InternalAPI.reinit!(
        cache::EisenstatWalkerForcing2Cache; p = cache.p, kwargs...
)
    cache.p = p
end



"""
    EisenstatWalkerNewtonKrylov(;
        concrete_jac = nothing, linsolve = nothing, linesearch = missing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing
    )

An advanced Newton-Krylov implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems.
"""
function EisenstatWalkerNewtonKrylov(;
        concrete_jac = nothing, linsolve::LinearSolve.AbstractKrylovSubspaceMethod, linesearch = nothing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing, forcing = EisenstatWalkerForcing2(),
)
    return GeneralizedFirstOrderAlgorithm(;
        linesearch,
        descent = NewtonDescent(; linsolve),
        autodiff, vjp_autodiff, jvp_autodiff,
        concrete_jac,
        forcing,
        name = :EisenstatWalkerNewtonKrylov
    )
end
