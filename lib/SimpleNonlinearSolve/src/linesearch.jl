# This is a copy of the version in NonlinearSolve.jl. Temporarily kept here till we move
# line searches into a dedicated package.
@kwdef @concrete struct LiFukushimaLineSearch
    lambda_0 = 1
    beta = 0.5
    sigma_1 = 0.001
    sigma_2 = 0.001
    eta = 0.1
    rho = 0.1
    nan_maxiters = missing
    maxiters::Int = 100
end

@concrete mutable struct LiFukushimaLineSearchCache{T <: Union{Nothing, Int}}
    ϕ
    λ₀
    β
    σ₁
    σ₂
    η
    ρ
    α
    nan_maxiters::T
    maxiters::Int
end

@concrete struct StaticLiFukushimaLineSearchCache
    f
    p
    λ₀
    β
    σ₁
    σ₂
    η
    ρ
    maxiters::Int
end

(alg::LiFukushimaLineSearch)(prob, fu, u) = __generic_init(alg, prob, fu, u)
function (alg::LiFukushimaLineSearch)(prob, fu::Union{Number, SArray},
        u::Union{Number, SArray})
    (alg.nan_maxiters === missing || alg.nan_maxiters === nothing) &&
        return __static_init(alg, prob, fu, u)
    @warn "`LiFukushimaLineSearch` with NaN checking is not non-allocating" maxlog=1
    return __generic_init(alg, prob, fu, u)
end

function __generic_init(alg::LiFukushimaLineSearch, prob, fu, u)
    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    T = promote_type(eltype(fu), eltype(u))

    ϕ = @closure (u, δu, α) -> begin
        @bb @. u_cache = u + α * δu
        return NONLINEARSOLVE_DEFAULT_NORM(__eval_f(prob, fu_cache, u_cache))
    end

    nan_maxiters = ifelse(alg.nan_maxiters === missing, 5, alg.nan_maxiters)

    return LiFukushimaLineSearchCache(ϕ, T(alg.lambda_0), T(alg.beta), T(alg.sigma_1),
        T(alg.sigma_2), T(alg.eta), T(alg.rho), T(true), nan_maxiters, alg.maxiters)
end

function __static_init(alg::LiFukushimaLineSearch, prob, fu, u)
    T = promote_type(eltype(fu), eltype(u))
    return StaticLiFukushimaLineSearchCache(prob.f, prob.p, T(alg.lambda_0), T(alg.beta),
        T(alg.sigma_1), T(alg.sigma_2), T(alg.eta), T(alg.rho), alg.maxiters)
end

function (cache::LiFukushimaLineSearchCache)(u, δu)
    T = promote_type(eltype(u), eltype(δu))
    ϕ = @closure α -> cache.ϕ(u, δu, α)
    fx_norm = ϕ(T(0))

    # Non-Blocking exit if the norm is NaN or Inf
    DiffEqBase.NAN_CHECK(fx_norm) && return cache.α

    # Early Terminate based on Eq. 2.7
    du_norm = NONLINEARSOLVE_DEFAULT_NORM(δu)
    fxλ_norm = ϕ(cache.α)
    fxλ_norm ≤ cache.ρ * fx_norm - cache.σ₂ * du_norm^2 && return cache.α

    λ₂, λ₁ = cache.λ₀, cache.λ₀
    fxλp_norm = ϕ(λ₂)

    if cache.nan_maxiters !== nothing
        if DiffEqBase.NAN_CHECK(fxλp_norm)
            nan_converged = false
            for _ in 1:(cache.nan_maxiters)
                λ₁, λ₂ = λ₂, cache.β * λ₂
                fxλp_norm = ϕ(λ₂)
                nan_converged = DiffEqBase.NAN_CHECK(fxλp_norm)::Bool
                nan_converged && break
            end
            nan_converged || return cache.α
        end
    end

    for i in 1:(cache.maxiters)
        fxλp_norm = ϕ(λ₂)
        converged = fxλp_norm ≤ (1 + cache.η) * fx_norm - cache.σ₁ * λ₂^2 * du_norm^2
        converged && return λ₂
        λ₁, λ₂ = λ₂, cache.β * λ₂
    end

    return cache.α
end

function (cache::StaticLiFukushimaLineSearchCache)(u, δu)
    T = promote_type(eltype(u), eltype(δu))

    # Early Terminate based on Eq. 2.7
    fx_norm = NONLINEARSOLVE_DEFAULT_NORM(cache.f(u, cache.p))
    du_norm = NONLINEARSOLVE_DEFAULT_NORM(δu)
    fxλ_norm = NONLINEARSOLVE_DEFAULT_NORM(cache.f(u .+ δu, cache.p))
    fxλ_norm ≤ cache.ρ * fx_norm - cache.σ₂ * du_norm^2 && return T(true)

    λ₂, λ₁ = cache.λ₀, cache.λ₀

    for i in 1:(cache.maxiters)
        fxλp_norm = NONLINEARSOLVE_DEFAULT_NORM(cache.f(u .+ λ₂ .* δu, cache.p))
        converged = fxλp_norm ≤ (1 + cache.η) * fx_norm - cache.σ₁ * λ₂^2 * du_norm^2
        converged && return λ₂
        λ₁, λ₂ = λ₂, cache.β * λ₂
    end

    return T(true)
end
