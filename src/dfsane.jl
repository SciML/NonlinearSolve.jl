struct DFSane{T}#<:AbstractNonlinearSolveAlgorithm
    σₘᵢₙ::T 
    σₘₐₓ::T 
    σ₁::T
    M::Int
    γ::T
    τₘᵢₙ::T
    τₘₐₓ::T 
    nₑₓₚ::Int
    # ηₛ::F = (f₍ₙₒᵣₘ₎₁, n, xₙ, fₙ) -> f₍ₙₒᵣₘ₎₁ ./ n .^ 2 # Would this change ever?
    max_inner_iterations::Int
end

function DFSane(;
    σₘᵢₙ = 1.0f-10,
    σₘₐₓ = 1.0f+10,
    σ₁ = 1.0f0,
    M = 10,
    γ = 1.0f-4,
    τₘᵢₙ = 0.1f0,
    τₘₐₓ = 0.5f0,
    nₑₓₚ= 2,
    #ηₛ::F = (f₍ₙₒᵣₘ₎₁, n, xₙ, fₙ) -> f₍ₙₒᵣₘ₎₁ ./ n .^ 2
    max_inner_iterations = 1000)

    return DFSane{typeof(σₘᵢₙ)}(σₘᵢₙ, # Typeof thing?
    σₘₐₓ,
    σ₁,
    M,
    γ,
    τₘᵢₙ,
    τₘₐₓ,
    nₑₓₚ,
    #ηₛ::F = (f₍ₙₒᵣₘ₎₁, n, xₙ, fₙ) -> f₍ₙₒᵣₘ₎₁ ./ n .^ 2
    max_inner_iterations)
end
mutable struct DFSaneCache{iip}
    f::fType
    alg::algType
    uₙ::uType
    uₙ₋₁::uType
    fuₙ::resType
    fuₙ₋₁::resType
    f₍ₙₒᵣₘ₎ₙ::resType
    f̄::resType
    ff::Function
    p::pType
    force_stop::Bool
    maxiters::Int
    internalnorm::INType
    retcode::SciMLBase.ReturnCode.T
    abstol::tolType
    prob::probType
    stats::NLStats
    σₙ::σₙType
    σₘᵢₙ::σType
    σₘₐₓ::σType
    σ_sign::σType
    α₁::α₁Type
    α₋::αType
    α₊::αType
    𝒹::𝒹Type
    ℋ::ℋType
    η::ηType
    𝒸::𝒸Type
    N::NType
    function DFSaneCache()
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
        u = prob.u0
    else
        u = deepcopy(prob.u0)
    end
    f = prob.f
    p = prob.p
    if iip
        fu = zero(u)
        f(fu, u, p)
    else
        fu = f(u, p)
    end

    return DFSaneCache{iip}(f, alg, u, fu, p, false, maxiters, internalnorm,
                            ReturnCode.Default, abstol, prob, NLStats(1, 0, 0, 0, 0)) # What should NL stats be?
end

function perform_step!(cache::DFSaneCache{true})
    @unpack σₙ, σₘᵢₙ, σₘₐₓ, 𝒹, fuₙ₋₁, fuₙ,
    uₙ₋₁, f̄, ℋ, α₊, α₁, α₋, uₙ, η, ff, f₍ₙₒᵣₘ₎ₙ,f₍ₙₒᵣₘ₎₋₁, γ, N, = cache

    T = eltype(uₙ)
    n = cache.stats.nsteps
    # Spectral parameter range check
    @. σₙ = sign(σₙ) * clamp(abs(σₙ), σₘᵢₙ, σₘₐₓ)

    # Line search direction
    @. 𝒹 = -σₙ * fuₙ₋₁

    @. η = f₍ₙₒᵣₘ₎₋₁ / n^2 # Will rename initial norm

    maximum!(f̄, ℋ)
    fill!(α₊, α₁)
    fill!(α₋, α₁)
    @. uₙ = uₙ₋₁ + α₊ * 𝒹

    ff(fuₙ, f₍ₙₒᵣₘ₎ₙ, uₙ)

    for _ in 1:(cache.max_inner_iterations)
       @. 𝒸 = f̄ + η - γ * α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁

        (sum(f₍ₙₒᵣₘ₎ₙ .≤ 𝒸) ≥ N ÷ 2) && break

       @. α₊ = clamp(α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁ /
                             (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₊ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
                             τₘᵢₙ * α₊,
                             τₘₐₓ * α₊)

        @. uₙ = uₙ₋₁ - α₋ * 𝒹
        ff(fuₙ, f₍ₙₒᵣₘ₎ₙ, uₙ)

        (sum(f₍ₙₒᵣₘ₎ₙ .≤ 𝒸) ≥ N ÷ 2) && break

        @. α₋ = clamp(α₋^2 * f₍ₙₒᵣₘ₎ₙ₋₁ / (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₋ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
                      τₘᵢₙ * α₋,
                      τₘₐₓ * α₋)
        @. uₙ = uₙ₋₁ + α₊ * 𝒹
        ff(fuₙ, f₍ₙₒᵣₘ₎ₙ, uₙ)
    end

    if cache.internalnorm(cache.fuₙ) < cache.abstol
        cache.force_stop = true
    end

    # Update spectral parameter
    @. u₋₁ = u - u₋₁
    @. fu₋₁ = fu - fu₋₁

    sum!(abs2, α₊, u₋₁)
    sum!(α₋, u₋₁ .* fu₋₁)
    σₙ .= α₊ ./ (α₋ .+ T(1e-5))

    # Take step
    @. u₋₁ = u
    @. fu₋₁ = fu
    @. f₍ₙₒᵣₘ₎ₙ₋₁ = f₍ₙₒᵣₘ₎ₙ

    # Update history
    ℋ[n % M + 1, :] .= view(f₍ₙₒᵣₘ₎ₙ, 1, :)
    cache.stats.nf += 1
    return nothing
end

function perform_step!(cache::DFSaneCache{false})
    return nothing
end

function SciMLBase.solve!(cache::DFSaneCache)
    while !cache.force_stop && cache.stats.nsteps < cache.maxiters
        perform_step!(cache)
        cache.stats.nsteps += 1
    end

    if cache.stats.nsteps == cache.maxiters
        cache.retcode = ReturnCode.MaxIters
    else
        cache.retcode = ReturnCode.Success
    end

    SciMLBase.build_solution(cache.prob, cache.alg, cache.u, cache.fu;
                             retcode = cache.retcode, stats = cache.stats)
end
