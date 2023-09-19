struct DFSane{T, F} <: AbstractNonlinearSolveAlgorithm
    σₘᵢₙ::T
    σₘₐₓ::T
    σ₁::T
    M::Int
    γ::T
    τₘᵢₙ::T
    τₘₐₓ::T
    nₑₓₚ::Int
    ηₛ::F
    max_inner_iterations::Int
end

function DFSane(; σₘᵢₙ = 1e-10,
                σₘₐₓ = 1e+10,
                σ₁ = 1.0,
                M = 10,
                γ = 1e-4,
                τₘᵢₙ = 0.1,
                τₘₐₓ = 0.5,
                nₑₓₚ = 2,
                ηₛ = (f₍ₙₒᵣₘ₎₁, n, xₙ, fₙ) -> f₍ₙₒᵣₘ₎₁ / n^2,
                max_inner_iterations = 1000)
    return DFSane{typeof(σₘᵢₙ), typeof(ηₛ)}(σₘᵢₙ,
                                            σₘₐₓ,
                                            σ₁,
                                            M,
                                            γ,
                                            τₘᵢₙ,
                                            τₘₐₓ,
                                            nₑₓₚ,
                                            ηₛ,
                                            max_inner_iterations)
end
mutable struct DFSaneCache{iip, fType, algType, uType, resType, T, pType,
                           INType,
                           tolType,
                           probType}
    f::fType
    alg::algType
    uₙ::uType
    uₙ₋₁::uType
    fuₙ::resType
    fuₙ₋₁::resType
    𝒹::uType
    ℋ::uType
    f₍ₙₒᵣₘ₎ₙ₋₁::T
    f₍ₙₒᵣₘ₎₀::T
    M::Int
    σₙ::T
    σₘᵢₙ::T
    σₘₐₓ::T
    α₁::T
    γ::T
    τₘᵢₙ::T
    τₘₐₓ::T
    nₑₓₚ::Int
    p::pType
    force_stop::Bool
    maxiters::Int
    internalnorm::INType
    retcode::SciMLBase.ReturnCode.T
    abstol::tolType
    prob::probType
    stats::NLStats
    function DFSaneCache{iip}(f::fType, alg::algType, uₙ::uType, uₙ₋₁::uType,
                              fuₙ::resType, fuₙ₋₁::resType, 𝒹::uType, ℋ::uType,
                              f₍ₙₒᵣₘ₎ₙ₋₁::T, f₍ₙₒᵣₘ₎₀::T, M::Int, σₙ::T, σₘᵢₙ::T, σₘₐₓ::T,
                              α₁::T, γ::T, τₘᵢₙ::T, τₘₐₓ::T, nₑₓₚ::Int, p::pType,
                              force_stop::Bool, maxiters::Int, internalnorm::INType,
                              retcode::SciMLBase.ReturnCode.T, abstol::tolType,
                              prob::probType,
                              stats::NLStats) where {iip, fType, algType, uType,
                                                     resType, T, pType, INType,
                                                     tolType,
                                                     probType
                                                     }
        new{iip, fType, algType, uType, resType, T, pType, INType, tolType,
            probType
            }(f, alg, uₙ, uₙ₋₁, fuₙ, fuₙ₋₁, 𝒹, ℋ, f₍ₙₒᵣₘ₎ₙ₋₁, f₍ₙₒᵣₘ₎₀, M, σₙ,
              σₘᵢₙ, σₘₐₓ, α₁, γ, τₘᵢₙ,
              τₘₐₓ, nₑₓₚ, p, force_stop, maxiters, internalnorm,
              retcode,
              abstol, prob, stats)
    end
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::DFSane,
                          args...;
                          alias_u0 = false,
                          maxiters = 1000,
                          abstol = 1e-6,
                          internalnorm = DEFAULT_NORM,
                          kwargs...) where {uType, iip}
    if alias_u0
        uₙ = prob.u0
    else
        uₙ = deepcopy(prob.u0)
    end

    p = prob.p
    T = eltype(uₙ)
    σₘᵢₙ, σₘₐₓ, γ, τₘᵢₙ, τₘₐₓ = T(alg.σₘᵢₙ), T(alg.σₘₐₓ), T(alg.γ), T(alg.τₘᵢₙ), T(alg.τₘₐₓ)
    α₁ = one(T)
    γ = T(alg.γ)
    f₍ₙₒᵣₘ₎ₙ₋₁ = α₁
    σₙ = T(alg.σ₁)
    M = alg.M
    nₑₓₚ = alg.nₑₓₚ
    𝒹, uₙ₋₁, fuₙ, fuₙ₋₁ = copy(uₙ), copy(uₙ), copy(uₙ), copy(uₙ)

    if iip
        f(dx, x) = prob.f(dx, x, p)
        f(fuₙ₋₁, uₙ₋₁)

    else
        f(x) = prob.f(x, p)
        fuₙ₋₁ = f(uₙ₋₁)
    end

    f₍ₙₒᵣₘ₎ₙ₋₁ = norm(fuₙ₋₁)^nₑₓₚ
    f₍ₙₒᵣₘ₎₀ = f₍ₙₒᵣₘ₎ₙ₋₁

    ℋ = fill(f₍ₙₒᵣₘ₎ₙ₋₁, M)

    return DFSaneCache{iip}(f, alg, uₙ, uₙ₋₁, fuₙ, fuₙ₋₁, 𝒹, ℋ, f₍ₙₒᵣₘ₎ₙ₋₁, f₍ₙₒᵣₘ₎₀,
                            M, σₙ, σₘᵢₙ, σₘₐₓ, α₁, γ, τₘᵢₙ,
                            τₘₐₓ, nₑₓₚ, p, false, maxiters,
                            internalnorm, ReturnCode.Default, abstol, prob,
                            NLStats(1, 0, 0, 0, 0))
end

function perform_step!(cache::DFSaneCache{true})
    @unpack f, alg, f₍ₙₒᵣₘ₎ₙ₋₁, f₍ₙₒᵣₘ₎₀,
    σₙ, σₘᵢₙ, σₘₐₓ, α₁, γ, τₘᵢₙ, τₘₐₓ, nₑₓₚ, M = cache

    T = eltype(cache.uₙ)
    n = cache.stats.nsteps

    # Spectral parameter range check
    σₙ = sign(σₙ) * clamp(abs(σₙ), σₘᵢₙ, σₘₐₓ)

    # Line search direction
    @. cache.𝒹 = -σₙ * cache.fuₙ₋₁

    η = alg.ηₛ(f₍ₙₒᵣₘ₎₀, n, cache.uₙ₋₁, cache.fuₙ₋₁)

    f̄ = maximum(cache.ℋ)
    α₊ = α₁
    α₋ = α₁
    @. cache.uₙ = cache.uₙ₋₁ + α₊ * cache.𝒹

    f(cache.fuₙ, cache.uₙ)
    f₍ₙₒᵣₘ₎ₙ = norm(cache.fuₙ)^nₑₓₚ
    for _ in 1:(cache.alg.max_inner_iterations)
        𝒸 = f̄ + η - γ * α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁

        f₍ₙₒᵣₘ₎ₙ ≤ 𝒸 && break

        α₊ = clamp(α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁ /
                   (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₊ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
                   τₘᵢₙ * α₊,
                   τₘₐₓ * α₊)
        @. cache.uₙ = cache.uₙ₋₁ - α₋ * cache.𝒹

        f(cache.fuₙ, cache.uₙ)
        f₍ₙₒᵣₘ₎ₙ = norm(cache.fuₙ)^nₑₓₚ

        f₍ₙₒᵣₘ₎ₙ .≤ 𝒸 && break

        α₋ = clamp(α₋^2 * f₍ₙₒᵣₘ₎ₙ₋₁ / (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₋ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
                   τₘᵢₙ * α₋,
                   τₘₐₓ * α₋)

        @. cache.uₙ = cache.uₙ₋₁ + α₊ * cache.𝒹
        f(cache.fuₙ, cache.uₙ)
        f₍ₙₒᵣₘ₎ₙ = norm(cache.fuₙ)^nₑₓₚ
    end

    if cache.internalnorm(cache.fuₙ) < cache.abstol
        cache.force_stop = true
    end

    # Update spectral parameter
    @. cache.uₙ₋₁ = cache.uₙ - cache.uₙ₋₁
    @. cache.fuₙ₋₁ = cache.fuₙ - cache.fuₙ₋₁

    α₊ = sum(abs2, cache.uₙ₋₁)
    @. cache.uₙ₋₁ = cache.uₙ₋₁ * cache.fuₙ₋₁
    α₋ = sum(cache.uₙ₋₁)
    cache.σₙ = α₊ / α₋

    # Spectral parameter bounds check
    if abs(cache.σₙ) > σₘₐₓ || abs(cache.σₙ) < σₘᵢₙ
        test_norm = sqrt(sum(abs2, cache.fuₙ₋₁))
        if test_norm > 1
            cache.σₙ = 1.0
        elseif testnorm < 1e-5
            cache.σₙ = 1e5
        else
            cache.σₙ = 1.0 / test_norm
        end
    end

    # Take step
    @. cache.uₙ₋₁ = cache.uₙ
    @. cache.fuₙ₋₁ = cache.fuₙ
    cache.f₍ₙₒᵣₘ₎ₙ₋₁ = f₍ₙₒᵣₘ₎ₙ

    # Update history
    cache.ℋ[n % M + 1] = f₍ₙₒᵣₘ₎ₙ
    cache.stats.nf += 1
    return nothing
end

function perform_step!(cache::DFSaneCache{false})
    @unpack f, alg, f₍ₙₒᵣₘ₎ₙ₋₁, f₍ₙₒᵣₘ₎₀,
    σₙ, σₘᵢₙ, σₘₐₓ, α₁, γ, τₘᵢₙ, τₘₐₓ, nₑₓₚ, M = cache

    T = eltype(cache.uₙ)
    n = cache.stats.nsteps

    # Spectral parameter range check
    σₙ = sign(σₙ) * clamp(abs(σₙ), σₘᵢₙ, σₘₐₓ)

    # Line search direction
    @. cache.𝒹 = -σₙ * cache.fuₙ₋₁

    η = alg.ηₛ(f₍ₙₒᵣₘ₎₀, n, cache.uₙ₋₁, cache.fuₙ₋₁)

    f̄ = maximum(cache.ℋ)
    α₊ = α₁
    α₋ = α₁
    @. cache.uₙ = cache.uₙ₋₁ + α₊ * cache.𝒹

    @. cache.fuₙ = f(cache.uₙ)
    f₍ₙₒᵣₘ₎ₙ = norm(cache.fuₙ)^nₑₓₚ

    for _ in 1:(cache.alg.max_inner_iterations)
        𝒸 = f̄ + η - γ * α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁

        f₍ₙₒᵣₘ₎ₙ ≤ 𝒸 && break

        α₊ = clamp(α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁ /
                   (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₊ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
                   τₘᵢₙ * α₊,
                   τₘₐₓ * α₊)
        @. cache.uₙ = cache.uₙ₋₁ - α₋ * cache.𝒹 # correct order?

        @. cache.fuₙ = f(cache.uₙ)
        f₍ₙₒᵣₘ₎ₙ = norm(cache.fuₙ)^nₑₓₚ

        (f₍ₙₒᵣₘ₎ₙ .≤ 𝒸) && break

        α₋ = clamp(α₋^2 * f₍ₙₒᵣₘ₎ₙ₋₁ / (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₋ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
                   τₘᵢₙ * α₋,
                   τₘₐₓ * α₋)

        @. cache.uₙ = cache.uₙ₋₁ + α₊ * cache.𝒹 # correct order?
        @. cache.fuₙ = f(cache.uₙ)
        f₍ₙₒᵣₘ₎ₙ = norm(cache.fuₙ)^nₑₓₚ
    end

    if cache.internalnorm(cache.fuₙ) < cache.abstol
        cache.force_stop = true
    end

    # Update spectral parameter
    @. cache.uₙ₋₁ = cache.uₙ - cache.uₙ₋₁
    @. cache.fuₙ₋₁ = cache.fuₙ - cache.fuₙ₋₁

    α₊ = sum(abs2, cache.uₙ₋₁)
    @. cache.uₙ₋₁ = cache.uₙ₋₁ * cache.fuₙ₋₁
    α₋ = sum(cache.uₙ₋₁)
    cache.σₙ = α₊ / α₋

    # Spectral parameter bounds check
    if abs(cache.σₙ) > σₘₐₓ || abs(cache.σₙ) < σₘᵢₙ
        test_norm = sqrt(sum(abs2, cache.fuₙ₋₁))
        if test_norm > 1
            cache.σₙ = 1.0
        elseif testnorm < 1e-5
            cache.σₙ = 1e5
        else
            cache.σₙ = 1.0 / test_norm
        end
    end

    # Take step
    @. cache.uₙ₋₁ = cache.uₙ
    @. cache.fuₙ₋₁ = cache.fuₙ
    cache.f₍ₙₒᵣₘ₎ₙ₋₁ = f₍ₙₒᵣₘ₎ₙ

    # Update history
    cache.ℋ[n % M + 1] = f₍ₙₒᵣₘ₎ₙ
    cache.stats.nf += 1
    return nothing
end

function SciMLBase.solve!(cache::DFSaneCache)
    while !cache.force_stop && cache.stats.nsteps < cache.maxiters
        cache.stats.nsteps += 1
        perform_step!(cache)
    end

    if cache.stats.nsteps == cache.maxiters
        cache.retcode = ReturnCode.MaxIters
    else
        cache.retcode = ReturnCode.Success
    end

    SciMLBase.build_solution(cache.prob, cache.alg, cache.uₙ, cache.fuₙ;
                             retcode = cache.retcode, stats = cache.stats)
end
