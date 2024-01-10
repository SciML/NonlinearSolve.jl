# This is a copy of the version in NonlinearSolve.jl. Temporarily kept here till we move
# line searches into a dedicated package. Renamed to `__` to avoid conflicts.
@kwdef @concrete struct __LiFukushimaLineSearch
    lambda_0 = 1
    beta = 1 // 2
    sigma_1 = 1 // 1000
    sigma_2 = 1 // 1000
    eta = 1 // 10
    rho = 9 // 10
    nan_maxiters::Int = 5
    maxiters::Int = 100
end

@concrete mutable struct __LiFukushimaLineSearchCache
    ϕ
    λ₀
    β
    σ₁
    σ₂
    η
    ρ
    α
    nan_maxiters::Int
    maxiters::Int
end

function (alg::__LiFukushimaLineSearch)(prob, fu, u)
    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    T = promote_type(eltype(fu), eltype(u))

    ϕ = @closure (u, δu, α) -> begin
        @bb @. u_cache = u + α * δu
        return norm(__eval_f(prob, fu_cache, u_cache), 2)
    end

    return __LiFukushimaLineSearchCache(ϕ, T(alg.lambda_0), T(alg.beta), T(alg.sigma_1),
        T(alg.sigma_2), T(alg.eta), T(alg.rho), T(true), alg.nan_maxiters, alg.maxiters)
end

function (cache::__LiFukushimaLineSearchCache)(u, δu)
    T = promote_type(eltype(u), eltype(δu))
    ϕ = @closure α -> cache.ϕ(u, δu, α)

    fx_norm = ϕ(T(0))

    # Non-Blocking exit if the norm is NaN or Inf
    !isfinite(fx_norm) && return cache.α

    # Early Terminate based on Eq. 2.7
    du_norm = norm(δu, 2)
    fxλ_norm = ϕ(cache.α)
    fxλ_norm ≤ cache.ρ * fx_norm - cache.σ₂ * du_norm^2 && return cache.α

    λ₂, λ₁ = cache.λ₀, cache.λ₀
    fxλp_norm = ϕ(λ₂)

    if !isfinite(fxλp_norm)
        nan_converged = false
        for _ in 1:(cache.nan_maxiters)
            λ₁, λ₂ = λ₂, cache.β * λ₂
            fxλp_norm = ϕ(λ₂)
            nan_converged = isfinite(fxλp_norm)
            nan_converged && break
        end
        nan_converged || return cache.α
    end

    for i in 1:(cache.maxiters)
        fxλp_norm = ϕ(λ₂)
        converged = fxλp_norm ≤ (1 + cache.η) * fx_norm - cache.σ₁ * λ₂^2 * du_norm^2
        converged && return λ₂
        λ₁, λ₂ = λ₂, cache.β * λ₂
    end

    return cache.α
end
