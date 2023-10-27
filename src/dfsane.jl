"""
    DFSane(; σ_min::Real = 1e-10, σ_max::Real = 1e10, σ_1::Real = 1.0,
        M::Int = 10, γ::Real = 1e-4, τ_min::Real = 0.1, τ_max::Real = 0.5,
        n_exp::Int = 2, η_strategy::Function = (fn_1, n, x_n, f_n) -> fn_1 / n^2,
        max_inner_iterations::Int = 1000)

A low-overhead and allocation-free implementation of the df-sane method for solving large-scale nonlinear
systems of equations. For in depth information about all the parameters and the algorithm,
see the paper: [W LaCruz, JM Martinez, and M Raydan (2006), Spectral residual mathod without
gradient information for solving large-scale nonlinear systems of equations, Mathematics of
Computation, 75, 1429-1448.](https://www.researchgate.net/publication/220576479_Spectral_Residual_Method_without_Gradient_Information_for_Solving_Large-Scale_Nonlinear_Systems_of_Equations)

See also the implementation in [SimpleNonlinearSolve.jl](https://github.com/SciML/SimpleNonlinearSolve.jl/blob/main/src/dfsane.jl)

### Keyword Arguments

- `σ_min`: the minimum value of the spectral coefficient `σₙ` which is related to the step
  size in the algorithm. Defaults to `1e-10`.
- `σ_max`: the maximum value of the spectral coefficient `σₙ` which is related to the step
  size in the algorithm. Defaults to `1e10`.
- `σ_1`: the initial value of the spectral coefficient `σₙ` which is related to the step
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
- `τ_min`: if a step is rejected the new step size will get multiplied by factor, and this
  parameter is the minimum value of that factor. Defaults to `0.1`.
- `τ_max`: if a step is rejected the new step size will get multiplied by factor, and this
  parameter is the maximum value of that factor. Defaults to `0.5`.
- `n_exp`: the exponent of the loss, i.e. ``f_n=||F(x_n)||^{n_exp}``. The paper uses
  `n_exp ∈ {1,2}`. Defaults to `2`.
- `η_strategy`:  function to determine the parameter `η`, which enables growth
  of ``||f_n||^2``. Called as ``η = η_strategy(fn_1, n, x_n, f_n)`` with `fn_1` initialized as
  ``fn_1=||f(x_1)||^{n_exp}``, `n` is the iteration number, `x_n` is the current `x`-value and
  `f_n` the current residual. Should satisfy ``η > 0`` and ``∑ₖ ηₖ < ∞``. Defaults to
  ``fn_1 / n^2``.
- `max_inner_iterations`: the maximum number of iterations allowed for the inner loop of the
  algorithm. Defaults to `1000`.
"""
struct DFSane{T, F} <: AbstractNonlinearSolveAlgorithm
    σ_min::T
    σ_max::T
    σ_1::T
    M::Int
    γ::T
    τ_min::T
    τ_max::T
    n_exp::Int
    η_strategy::F
    max_inner_iterations::Int
end

function DFSane(; σ_min = 1e-10, σ_max = 1e+10, σ_1 = 1.0, M = 10, γ = 1e-4, τ_min = 0.1,
    τ_max = 0.5, n_exp = 2, η_strategy = (fn_1, n, x_n, f_n) -> fn_1 / n^2,
    max_inner_iterations = 1000)
    return DFSane{typeof(σ_min), typeof(η_strategy)}(σ_min, σ_max, σ_1, M, γ, τ_min, τ_max,
        n_exp, η_strategy, max_inner_iterations)
end

@concrete mutable struct DFSaneCache{iip}
    alg
    uₙ
    uₙ₋₁
    fuₙ
    fuₙ₋₁
    𝒹
    ℋ
    f₍ₙₒᵣₘ₎ₙ₋₁
    f₍ₙₒᵣₘ₎₀
    M
    σₙ
    σₘᵢₙ
    σₘₐₓ
    α₁
    γ
    τₘᵢₙ
    τₘₐₓ
    nₑₓₚ::Int
    p
    force_stop::Bool
    maxiters::Int
    internalnorm
    retcode::SciMLBase.ReturnCode.T
    abstol
    reltol
    prob
    stats::NLStats
    termination_condition
    tc_storage
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::DFSane, args...;
    alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
    termination_condition = nothing, internalnorm = DEFAULT_NORM,
    kwargs...) where {uType, iip}
    uₙ = alias_u0 ? prob.u0 : deepcopy(prob.u0)

    p = prob.p
    T = eltype(uₙ)
    σₘᵢₙ, σₘₐₓ, γ, τₘᵢₙ, τₘₐₓ = T(alg.σ_min), T(alg.σ_max), T(alg.γ), T(alg.τ_min),
    T(alg.τ_max)
    α₁ = one(T)
    γ = T(alg.γ)
    f₍ₙₒᵣₘ₎ₙ₋₁ = α₁
    σₙ = T(alg.σ_1)
    M = alg.M
    nₑₓₚ = alg.n_exp
    𝒹, uₙ₋₁, fuₙ, fuₙ₋₁ = copy(uₙ), copy(uₙ), copy(uₙ), copy(uₙ)

    if iip
        prob.f(fuₙ₋₁, uₙ₋₁, p)
    else
        fuₙ₋₁ = prob.f(uₙ₋₁, p)
    end

    f₍ₙₒᵣₘ₎ₙ₋₁ = norm(fuₙ₋₁)^nₑₓₚ
    f₍ₙₒᵣₘ₎₀ = f₍ₙₒᵣₘ₎ₙ₋₁

    ℋ = fill(f₍ₙₒᵣₘ₎ₙ₋₁, M)

    abstol, reltol, termination_condition = _init_termination_elements(abstol, reltol,
        termination_condition, T)

    mode = DiffEqBase.get_termination_mode(termination_condition)

    storage = mode ∈ DiffEqBase.SAFE_TERMINATION_MODES ? NLSolveSafeTerminationResult() :
              nothing

    return DFSaneCache{iip}(alg, uₙ, uₙ₋₁, fuₙ, fuₙ₋₁, 𝒹, ℋ, f₍ₙₒᵣₘ₎ₙ₋₁, f₍ₙₒᵣₘ₎₀,
        M, σₙ, σₘᵢₙ, σₘₐₓ, α₁, γ, τₘᵢₙ, τₘₐₓ, nₑₓₚ, p, false, maxiters,
        internalnorm, ReturnCode.Default, abstol, reltol, prob, NLStats(1, 0, 0, 0, 0),
        termination_condition, storage)
end

function perform_step!(cache::DFSaneCache{true})
    @unpack alg, f₍ₙₒᵣₘ₎ₙ₋₁, f₍ₙₒᵣₘ₎₀, σₙ, σₘᵢₙ, σₘₐₓ, α₁, γ, τₘᵢₙ, τₘₐₓ, nₑₓₚ, M, tc_storage = cache

    termination_condition = cache.termination_condition(tc_storage)
    f = (dx, x) -> cache.prob.f(dx, x, cache.p)

    T = eltype(cache.uₙ)
    n = cache.stats.nsteps

    # Spectral parameter range check
    σₙ = sign(σₙ) * clamp(abs(σₙ), σₘᵢₙ, σₘₐₓ)

    # Line search direction
    @. cache.𝒹 = -σₙ * cache.fuₙ₋₁

    η = alg.η_strategy(f₍ₙₒᵣₘ₎₀, n, cache.uₙ₋₁, cache.fuₙ₋₁)

    f̄ = maximum(cache.ℋ)
    α₊ = α₁
    α₋ = α₁
    @. cache.uₙ = cache.uₙ₋₁ + α₊ * cache.𝒹

    f(cache.fuₙ, cache.uₙ)
    f₍ₙₒᵣₘ₎ₙ = norm(cache.fuₙ)^nₑₓₚ
    for jjj in 1:(cache.alg.max_inner_iterations)
        𝒸 = f̄ + η - γ * α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁

        f₍ₙₒᵣₘ₎ₙ ≤ 𝒸 && break

        α₊ = clamp(α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁ / (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₊ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
            τₘᵢₙ * α₊, τₘₐₓ * α₊)
        @. cache.uₙ = cache.uₙ₋₁ - α₋ * cache.𝒹

        f(cache.fuₙ, cache.uₙ)
        f₍ₙₒᵣₘ₎ₙ = norm(cache.fuₙ)^nₑₓₚ

        f₍ₙₒᵣₘ₎ₙ .≤ 𝒸 && break

        α₋ = clamp(α₋^2 * f₍ₙₒᵣₘ₎ₙ₋₁ / (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₋ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
            τₘᵢₙ * α₋, τₘₐₓ * α₋)

        @. cache.uₙ = cache.uₙ₋₁ + α₊ * cache.𝒹
        f(cache.fuₙ, cache.uₙ)
        f₍ₙₒᵣₘ₎ₙ = norm(cache.fuₙ)^nₑₓₚ
    end

    if termination_condition(cache.fuₙ, cache.uₙ, cache.uₙ₋₁, cache.abstol, cache.reltol)
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
        cache.σₙ = clamp(T(1) / test_norm, T(1), T(1e5))
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
    @unpack alg, f₍ₙₒᵣₘ₎ₙ₋₁, f₍ₙₒᵣₘ₎₀, σₙ, σₘᵢₙ, σₘₐₓ, α₁, γ, τₘᵢₙ, τₘₐₓ, nₑₓₚ, M, tc_storage = cache

    termination_condition = cache.termination_condition(tc_storage)
    f = x -> cache.prob.f(x, cache.p)

    T = eltype(cache.uₙ)
    n = cache.stats.nsteps

    # Spectral parameter range check
    σₙ = sign(σₙ) * clamp(abs(σₙ), σₘᵢₙ, σₘₐₓ)

    # Line search direction
    cache.𝒹 = -σₙ * cache.fuₙ₋₁

    η = alg.η_strategy(f₍ₙₒᵣₘ₎₀, n, cache.uₙ₋₁, cache.fuₙ₋₁)

    f̄ = maximum(cache.ℋ)
    α₊ = α₁
    α₋ = α₁
    cache.uₙ = cache.uₙ₋₁ + α₊ * cache.𝒹

    cache.fuₙ = f(cache.uₙ)
    f₍ₙₒᵣₘ₎ₙ = norm(cache.fuₙ)^nₑₓₚ
    for _ in 1:(cache.alg.max_inner_iterations)
        𝒸 = f̄ + η - γ * α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁

        f₍ₙₒᵣₘ₎ₙ ≤ 𝒸 && break

        α₊ = clamp(α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁ / (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₊ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
            τₘᵢₙ * α₊, τₘₐₓ * α₊)
        cache.uₙ = @. cache.uₙ₋₁ - α₋ * cache.𝒹

        cache.fuₙ = f(cache.uₙ)
        f₍ₙₒᵣₘ₎ₙ = norm(cache.fuₙ)^nₑₓₚ

        f₍ₙₒᵣₘ₎ₙ .≤ 𝒸 && break

        α₋ = clamp(α₋^2 * f₍ₙₒᵣₘ₎ₙ₋₁ / (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₋ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
            τₘᵢₙ * α₋, τₘₐₓ * α₋)

        cache.uₙ = @. cache.uₙ₋₁ + α₊ * cache.𝒹
        cache.fuₙ = f(cache.uₙ)
        f₍ₙₒᵣₘ₎ₙ = norm(cache.fuₙ)^nₑₓₚ
    end

    if termination_condition(cache.fuₙ, cache.uₙ, cache.uₙ₋₁, cache.abstol, cache.reltol)
        cache.force_stop = true
    end

    # Update spectral parameter
    cache.uₙ₋₁ = @. cache.uₙ - cache.uₙ₋₁
    cache.fuₙ₋₁ = @. cache.fuₙ - cache.fuₙ₋₁

    α₊ = sum(abs2, cache.uₙ₋₁)
    cache.uₙ₋₁ = @. cache.uₙ₋₁ * cache.fuₙ₋₁
    α₋ = sum(cache.uₙ₋₁)
    cache.σₙ = α₊ / α₋

    # Spectral parameter bounds check
    if abs(cache.σₙ) > σₘₐₓ || abs(cache.σₙ) < σₘᵢₙ
        test_norm = sqrt(sum(abs2, cache.fuₙ₋₁))
        cache.σₙ = clamp(T(1) / test_norm, T(1), T(1e5))
    end

    # Take step
    cache.uₙ₋₁ = cache.uₙ
    cache.fuₙ₋₁ = cache.fuₙ
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

    return SciMLBase.build_solution(cache.prob, cache.alg, cache.uₙ, cache.fuₙ;
        retcode = cache.retcode, stats = cache.stats)
end

function SciMLBase.reinit!(cache::DFSaneCache{iip}, u0 = cache.uₙ; p = cache.p,
    abstol = cache.abstol, reltol = cache.reltol,
    termination_condition = cache.termination_condition,
    maxiters = cache.maxiters) where {iip}
    cache.p = p
    if iip
        recursivecopy!(cache.uₙ, u0)
        recursivecopy!(cache.uₙ₋₁, u0)
        cache.prob.f(cache.fuₙ, cache.uₙ, p)
        cache.prob.f(cache.fuₙ₋₁, cache.uₙ, p)
    else
        cache.uₙ = u0
        cache.uₙ₋₁ = u0
        cache.fuₙ = cache.prob.f(cache.uₙ, p)
        cache.fuₙ₋₁ = cache.prob.f(cache.uₙ, p)
    end

    cache.f₍ₙₒᵣₘ₎ₙ₋₁ = norm(cache.fuₙ₋₁)^cache.nₑₓₚ
    cache.f₍ₙₒᵣₘ₎₀ = cache.f₍ₙₒᵣₘ₎ₙ₋₁
    fill!(cache.ℋ, cache.f₍ₙₒᵣₘ₎ₙ₋₁)

    T = eltype(cache.uₙ)
    cache.σₙ = T(cache.alg.σ_1)

    termination_condition = _get_reinit_termination_condition(cache, abstol, reltol,
        termination_condition)

    cache.abstol = abstol
    cache.reltol = reltol
    cache.termination_condition = termination_condition
    cache.maxiters = maxiters
    cache.stats.nf = 1
    cache.stats.nsteps = 1
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
    return cache
end
