"""
    DFSane(; σₘᵢₙ::Real = 1e-10, σₘₐₓ::Real = 1e10, σ₁::Real = 1.0,
        M::Int = 10, γ::Real = 1e-4, τₘᵢₙ::Real = 0.1, τₘₐₓ::Real = 0.5,
        nₑₓₚ::Int = 2, ηₛ::Function = (f₍ₙₒᵣₘ₎₁, n, xₙ, fₙ) -> f₍ₙₒᵣₘ₎₁ / n^2,
        max_inner_iterations::Int = 1000)

A low-overhead and allocation-free implementation of the df-sane method for solving large-scale nonlinear
systems of equations. For in depth information about all the parameters and the algorithm,
see the paper: [W LaCruz, JM Martinez, and M Raydan (2006), Spectral residual mathod without
gradient information for solving large-scale nonlinear systems of equations, Mathematics of
Computation, 75, 1429-1448.](https://www.researchgate.net/publication/220576479_Spectral_Residual_Method_without_Gradient_Information_for_Solving_Large-Scale_Nonlinear_Systems_of_Equations)

See also the implementation in [SimpleNonlinearSolve.jl](https://github.com/SciML/SimpleNonlinearSolve.jl/blob/main/src/dfsane.jl)

### Keyword Arguments

- `σₘᵢₙ`: the minimum value of the spectral coefficient `σₙ` which is related to the step
  size in the algorithm. Defaults to `1e-10`.
- `σₘₐₓ`: the maximum value of the spectral coefficient `σₙ` which is related to the step
  size in the algorithm. Defaults to `1e10`.
- `σ₁`: the initial value of the spectral coefficient `σₙ` which is related to the step
  size in the algorithm.. Defaults to `1.0`.
- `M`: The monotonicity of the algorithm is determined by a this positive integer.
  A value of 1 for `M` would result in strict monotonicity in the decrease of the L2-norm
  of the function `f`. However, higher values allow for more flexibility in this reduction.
  Despite this, the algorithm still ensures global convergence through the use of a
  non-monotone line-search algorithm that adheres to the Grippo-Lampariello-Lucidi
  condition. Values in the range of 5 to 20 are usually sufficient, but some cases may call
  for a higher value of `M`. The default setting is 10.
- `γ`: a parameter that influences if a proposed step will be accepted. Higher value of `γ`
  will make the algorithm more restrictive in accepting steps. Defaults to `1e-4`.
- `τₘᵢₙ`: if a step is rejected the new step size will get multiplied by factor, and this
  parameter is the minimum value of that factor. Defaults to `0.1`.
- `τₘₐₓ`: if a step is rejected the new step size will get multiplied by factor, and this
  parameter is the maximum value of that factor. Defaults to `0.5`.
- `nₑₓₚ`: the exponent of the loss, i.e. ``fₙ=||F(xₙ)||^{nₑₓₚ}``. The paper uses
  `nₑₓₚ ∈ {1,2}`. Defaults to `2`.
- `ηₛ`:  function to determine the parameter `η`, which enables growth
  of ``||fₙ||^2``. Called as ``η = ηₛ(f₍ₙₒᵣₘ₎₁, n, xₙ, fₙ)`` with `f₍ₙₒᵣₘ₎₁` initialized as
  ``f₍ₙₒᵣₘ₎₁=||f(x₁)||^{nₑₓₚ}``, `n` is the iteration number, `xₙ` is the current `x`-value and
  `fₙ` the current residual. Should satisfy ``η > 0`` and ``∑ₖ ηₖ < ∞``. Defaults to
  ``f₍ₙₒᵣₘ₎₁ / n^2``.
- `max_inner_iterations`: the maximum number of iterations allowed for the inner loop of the
  algorithm. Defaults to `1000`.
"""
(f₍ₙₒᵣₘ₎₁, n, xₙ, fₙ) -> f₍ₙₒᵣₘ₎₁ / n^2
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
        f = (dx, x) -> prob.f(dx, x, p)
        f(fuₙ₋₁, uₙ₋₁)

    else
        f = (x) -> prob.f(x, p)
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
        elseif test_norm < 1e-5
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

    cache.fuₙ .= f(cache.uₙ)
    f₍ₙₒᵣₘ₎ₙ = norm(cache.fuₙ)^nₑₓₚ
    for _ in 1:(cache.alg.max_inner_iterations)
        𝒸 = f̄ + η - γ * α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁

        f₍ₙₒᵣₘ₎ₙ ≤ 𝒸 && break

        α₊ = clamp(α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁ /
                   (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₊ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
                   τₘᵢₙ * α₊,
                   τₘₐₓ * α₊)
        @. cache.uₙ = cache.uₙ₋₁ - α₋ * cache.𝒹

        cache.fuₙ .= f(cache.uₙ)
        f₍ₙₒᵣₘ₎ₙ = norm(cache.fuₙ)^nₑₓₚ

        f₍ₙₒᵣₘ₎ₙ .≤ 𝒸 && break

        α₋ = clamp(α₋^2 * f₍ₙₒᵣₘ₎ₙ₋₁ / (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₋ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
                   τₘᵢₙ * α₋,
                   τₘₐₓ * α₋)

        @. cache.uₙ = cache.uₙ₋₁ + α₊ * cache.𝒹
        cache.fuₙ .= f(cache.uₙ)
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
        elseif test_norm < 1e-5
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
