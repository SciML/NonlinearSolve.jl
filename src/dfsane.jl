mutable struct DFSaneCache{iip}
    f::fType
    alg::algType
    u::uType
    fu::resType
    p::pType
    force_stop::Bool
    maxiters::Int
    internalnorm::INType
    retcode::SciMLBase.ReturnCode.T
    abstol::tolType
    prob::probType
    stats::NLStats
    

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
                            ReturnCode.Default, abstol, prob, NLStats(1,0,0,0,0)) # What should NL stats be?
end

function perform_step!(cache::DFSaneCache{true})
    @unpack σₙ, σₘᵢₙ, σₘₐₓ, 𝒹, fₙ₋₁,fₙ, n,
    xₙ₋₁, f̄, ℋ, α₊, α₁, α₋, xₙ,η,ff!, f₍ₙₒᵣₘ₎ₙ, = cache

    # Spectral parameter range check
    @. σₙ = sign(σₙ) * clamp(abs(σₙ), σₘᵢₙ, σₘₐₓ)

    # Line search direction
    @. 𝒹 = -σₙ * fₙ₋₁

    η = ηₛ(n, xₙ₋₁, fₙ₋₁)
    maximum!(f̄, ℋ)
    fill!(α₊, α₁)
    fill!(α₋, α₁)
    @. xₙ = xₙ₋₁ + α₊ * 𝒹

    ff(fₙ, f₍ₙₒᵣₘ₎ₙ, xₙ)

    for _ in 1:(cache.max_inner_iterations)
        𝒸 = @. f̄ + η - γ * α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁

        (sum(f₍ₙₒᵣₘ₎ₙ .≤ 𝒸) ≥ N ÷ 2) && break

        @. α₊ = clamp(α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁ / (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₊ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
            τₘᵢₙ * α₊,
            τₘₐₓ * α₊)
        @. xₙ = xₙ₋₁ - α₋ * 𝒹
        ff(fₙ, f₍ₙₒᵣₘ₎ₙ, xₙ)

        (sum(f₍ₙₒᵣₘ₎ₙ .≤ 𝒸) ≥ N ÷ 2) && break

        @. α₋ = clamp(α₋^2 * f₍ₙₒᵣₘ₎ₙ₋₁ / (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₋ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
            τₘᵢₙ * α₋,
            τₘₐₓ * α₋)
        @. xₙ = xₙ₋₁ + α₊ * 𝒹
        ff(fₙ, f₍ₙₒᵣₘ₎ₙ, xₙ)
    end

    if cache.internalnorm(cache.fₙ) < cache.abstol
        cache.force_stop = true
    end

    # Update spectral parameter
    @. xₙ₋₁ = xₙ - xₙ₋₁
    @. fₙ₋₁ = fₙ - fₙ₋₁

    sum!(abs2, α₊, xₙ₋₁)
    sum!(α₋, xₙ₋₁ .* fₙ₋₁)
    σₙ .= α₊ ./ (α₋ .+ T(1e-5))

    # Take step
    @. xₙ₋₁ = xₙ
    @. fₙ₋₁ = fₙ
    @. f₍ₙₒᵣₘ₎ₙ₋₁ = f₍ₙₒᵣₘ₎ₙ

    # Update history
    ℋ[n % M + 1, :] .= view(f₍ₙₒᵣₘ₎ₙ, 1, :)
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
