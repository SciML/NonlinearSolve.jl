"""
    NoLineSearch <: AbstractNonlinearSolveLineSearchAlgorithm

Don't perform a line search. Just return the initial step length of `1`.
"""
struct NoLineSearch <: AbstractNonlinearSolveLineSearchAlgorithm end

@concrete mutable struct NoLineSearchCache <: AbstractNonlinearSolveLineSearchCache
    α
end

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::NoLineSearch, f::F, fu, u,
        p, args...; kwargs...) where {F}
    return NoLineSearchCache(promote_type(eltype(fu), eltype(u))(true))
end

SciMLBase.solve!(cache::NoLineSearchCache, u, du) = false, cache.α

"""
    LineSearchesJL(; method = LineSearches.Static(), autodiff = nothing, α = true)

Wrapper over algorithms from
[LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl/). Allows automatic
construction of the objective functions for the line search algorithms utilizing automatic
differentiation for fast Vector Jacobian Products.

### Arguments

  - `method`: the line search algorithm to use. Defaults to
    `method = LineSearches.Static()`, which means that the step size is fixed to the value
    of `alpha`.
  - `autodiff`: the automatic differentiation backend to use for the line search. Defaults
    to `AutoFiniteDiff()`, which means that finite differencing is used to compute the VJP.
    `AutoZygote()` will be faster in most cases, but it requires `Zygote.jl` to be manually
    installed and loaded.
  - `α`: the initial step size to use. Defaults to `true` (which is equivalent to `1`).
"""
@concrete struct LineSearchesJL <: AbstractNonlinearSolveLineSearchAlgorithm
    method
    initial_alpha
    autodiff
end

function Base.show(io::IO, alg::LineSearchesJL)
    str = "$(nameof(typeof(alg)))("
    modifiers = String[]
    __is_present(alg.autodiff) &&
        push!(modifiers, "autodiff = $(nameof(typeof(alg.autodiff)))()")
    alg.initial_alpha != true && push!(modifiers, "initial_alpha = $(alg.initial_alpha)")
    push!(modifiers, "method = $(nameof(typeof(alg.method)))()")
    print(io, str, join(modifiers, ", "), ")")
end

LineSearchesJL(method; kwargs...) = LineSearchesJL(; method, kwargs...)
function LineSearchesJL(; method = LineSearches.Static(), autodiff = nothing, α = true)
    return LineSearchesJL(method, α, autodiff)
end

Base.@deprecate_binding LineSearch LineSearchesJL true

# Wrapper over LineSearches.jl algorithms
@concrete mutable struct LineSearchesJLCache <: AbstractNonlinearSolveLineSearchCache
    ϕ
    dϕ
    ϕdϕ
    method
    alpha
    grad_op
    u_cache
    fu_cache
    nf::Base.RefValue{Int}
end

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::LineSearchesJL, f::F, fu, u,
        p, args...; internalnorm::IN = DEFAULT_NORM, kwargs...) where {F, IN}
    T = promote_type(eltype(fu), eltype(u))
    if u isa Number
        grad_op = @closure (u, fu) -> last(__value_derivative(Base.Fix2(f, p), u)) * fu
    else
        if SciMLBase.has_jvp(f)
            if isinplace(prob)
                g_cache = similar(u)
                grad_op = @closure (u, fu) -> f.vjp(g_cache, fu, u, p)
            else
                grad_op = @closure (u, fu) -> f.vjp(fu, u, p)
            end
        else
            autodiff = get_concrete_reverse_ad(alg.autodiff, prob;
                check_forward_mode = true)
            vjp_op = VecJacOperator(prob, fu, u; autodiff)
            if isinplace(prob)
                g_cache = similar(u)
                grad_op = @closure (u, fu) -> vjp_op(g_cache, fu, u, p)
            else
                grad_op = @closure (u, fu) -> vjp_op(fu, u, p)
            end
        end
    end

    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    nf = Base.RefValue(0)

    ϕ = @closure (u, du, α, u_cache, fu_cache) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(prob, fu_cache, u_cache, p)
        nf[] += 1
        return @fastmath internalnorm(fu_cache)^2 / 2
    end

    dϕ = @closure (u, du, α, u_cache, fu_cache, grad_op) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(prob, fu_cache, u_cache, p)
        nf[] += 1
        g₀ = grad_op(u_cache, fu_cache)
        return dot(g₀, du)
    end

    ϕdϕ = @closure (u, du, α, u_cache, fu_cache, grad_op) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(prob, fu_cache, u_cache, p)
        nf[] += 1
        g₀ = grad_op(u_cache, fu_cache)
        obj = @fastmath internalnorm(fu_cache)^2 / 2
        return obj, dot(g₀, du)
    end

    return LineSearchesJLCache(ϕ, dϕ, ϕdϕ, alg.method, T(alg.initial_alpha), grad_op,
        u_cache, fu_cache, nf)
end

function SciMLBase.solve!(cache::LineSearchesJLCache, u, du; kwargs...)
    ϕ = @closure α -> cache.ϕ(u, du, α, cache.u_cache, cache.fu_cache)
    dϕ = @closure α -> cache.dϕ(u, du, α, cache.u_cache, cache.fu_cache, cache.grad_op)
    ϕdϕ = @closure α -> cache.ϕdϕ(u, du, α, cache.u_cache, cache.fu_cache, cache.grad_op)

    ϕ₀, dϕ₀ = ϕdϕ(zero(eltype(u)))

    # Here we should be resetting the search direction for some algorithms especially
    # if we start mixing in jacobian reuse and such
    dϕ₀ ≥ 0 && return (false, one(eltype(u)))

    # We can technically reduce 1 axpy by reusing the returned value from cache.method
    # but it's not worth the extra complexity
    cache.alpha = first(cache.method(ϕ, dϕ, ϕdϕ, cache.alpha, ϕ₀, dϕ₀))
    return (true, cache.alpha)
end

"""
    RobustNonMonotoneLineSearch(; gamma = 1 // 10000, sigma_0 = 1)

Robust NonMonotone Line Search is a derivative free line search method from DF Sane.

### References

[1] La Cruz, William, José Martínez, and Marcos Raydan. "Spectral residual method without
gradient information for solving large-scale nonlinear systems of equations."
Mathematics of computation 75.255 (2006): 1429-1448.
"""
@kwdef @concrete struct RobustNonMonotoneLineSearch <:
                        AbstractNonlinearSolveLineSearchAlgorithm
    gamma = 1 // 10000
    sigma_1 = 1
    M::Int = 10
    tau_min = 1 // 10
    tau_max = 1 // 2
    n_exp::Int = 2
    maxiters::Int = 100
    η_strategy = (fn₁, n, uₙ, fₙ) -> fn₁ / n^2
end

@concrete mutable struct RobustNonMonotoneLineSearchCache <:
                         AbstractNonlinearSolveLineSearchCache
    ϕ
    u_cache
    fu_cache
    internalnorm
    maxiters::Int
    history
    γ
    σ₁
    M::Int
    τ_min
    τ_max
    nsteps::Int
    η_strategy
    n_exp::Int
    nf::Base.RefValue{Int}
end

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::RobustNonMonotoneLineSearch,
        f::F, fu, u, p, args...; internalnorm::IN = DEFAULT_NORM, kwargs...) where {F, IN}
    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    T = promote_type(eltype(fu), eltype(u))

    nf = Base.RefValue(0)
    ϕ = @closure (u, du, α, u_cache, fu_cache) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(prob, fu_cache, u_cache, p)
        nf[] += 1
        return internalnorm(fu_cache)^alg.n_exp
    end

    fn₁ = internalnorm(fu)^alg.n_exp
    η_strategy = @closure (n, xₙ, fₙ) -> alg.η_strategy(fn₁, n, xₙ, fₙ)

    return RobustNonMonotoneLineSearchCache(ϕ, u_cache, fu_cache, internalnorm,
        alg.maxiters, fill(fn₁, alg.M), T(alg.gamma), T(alg.sigma_1), alg.M, T(alg.tau_min),
        T(alg.tau_max), 0, η_strategy, alg.n_exp, nf)
end

function SciMLBase.solve!(cache::RobustNonMonotoneLineSearchCache, u, du; kwargs...)
    T = promote_type(eltype(u), eltype(du))
    ϕ = @closure α -> cache.ϕ(u, du, α, cache.u_cache, cache.fu_cache)
    f_norm_old = ϕ(eltype(u)(0))
    α₊, α₋ = T(cache.σ₁), T(cache.σ₁)
    η = cache.η_strategy(cache.nsteps, u, f_norm_old)
    f_bar = maximum(cache.history)

    for k in 1:(cache.maxiters)
        f_norm = ϕ(α₊)
        f_norm ≤ f_bar + η - cache.γ * α₊ * f_norm_old && return (false, α₊)

        α₊ *= clamp(α₊ * f_norm_old / (f_norm + (T(2) * α₊ - T(1)) * f_norm_old),
            cache.τ_min, cache.τ_max)

        f_norm = ϕ(-α₋)
        f_norm ≤ f_bar + η - cache.γ * α₋ * f_norm_old && return (false, -α₋)

        α₋ *= clamp(α₋ * f_norm_old / (f_norm + (T(2) * α₋ - T(1)) * f_norm_old),
            cache.τ_min, cache.τ_max)
    end

    return true, T(cache.σ₁)
end

function callback_into_cache!(topcache, cache::RobustNonMonotoneLineSearchCache, args...)
    fu = get_fu(topcache)
    cache.history[mod1(cache.nsteps, cache.M)] = cache.internalnorm(fu)^cache.n_exp
    cache.nsteps += 1
    return
end

"""
    LiFukushimaLineSearch(; lambda_0 = 1, beta = 1 // 2, sigma_1 = 1 // 1000,
        sigma_2 = 1 // 1000, eta = 1 // 10, nan_max_iter::Int = 5, maxiters::Int = 100)

A derivative-free line search and global convergence of Broyden-like method for nonlinear
equations by Dong-Hui Li & Masao Fukushima.

### References

[1] Li, Dong-Hui, and Masao Fukushima. "A derivative-free line search and global convergence
of Broyden-like method for nonlinear equations." Optimization methods and software 13.3
(2000): 181-201.
"""
@kwdef @concrete struct LiFukushimaLineSearch <: AbstractNonlinearSolveLineSearchAlgorithm
    lambda_0 = 1
    beta = 1 // 2
    sigma_1 = 1 // 1000
    sigma_2 = 1 // 1000
    eta = 1 // 10
    rho = 9 // 10
    nan_max_iter::Int = 5  # TODO (breaking): Change this to nan_maxiters for uniformity
    maxiters::Int = 100
end

@concrete mutable struct LiFukushimaLineSearchCache <: AbstractNonlinearSolveLineSearchCache
    ϕ
    f
    p
    internalnorm
    u_cache
    fu_cache
    λ₀
    β
    σ₁
    σ₂
    η
    ρ
    α
    nan_maxiters::Int
    maxiters::Int
    nf::Base.RefValue{Int}
end

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::LiFukushimaLineSearch,
        f::F, fu, u, p, args...; internalnorm::IN = DEFAULT_NORM, kwargs...) where {F, IN}
    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    T = promote_type(eltype(fu), eltype(u))

    nf = Base.RefValue(0)
    ϕ = @closure (u, du, α, u_cache, fu_cache) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(prob, fu_cache, u_cache, p)
        nf[] += 1
        return internalnorm(fu_cache)
    end

    return LiFukushimaLineSearchCache(ϕ, f, p, internalnorm, u_cache, fu_cache,
        T(alg.lambda_0), T(alg.beta), T(alg.sigma_1), T(alg.sigma_2), T(alg.eta),
        T(alg.rho), T(true), alg.nan_max_iter, alg.maxiters, nf)
end

function SciMLBase.solve!(cache::LiFukushimaLineSearchCache, u, du; kwargs...)
    T = promote_type(eltype(u), eltype(du))
    ϕ = @closure α -> cache.ϕ(u, du, α, cache.u_cache, cache.fu_cache)

    fx_norm = ϕ(T(0))

    # Non-Blocking exit if the norm is NaN or Inf
    !isfinite(fx_norm) && return (true, cache.α)

    # Early Terminate based on Eq. 2.7
    du_norm = cache.internalnorm(du)
    fxλ_norm = ϕ(cache.α)
    fxλ_norm ≤ cache.ρ * fx_norm - cache.σ₂ * du_norm^2 && return (false, cache.α)

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
        nan_converged || return (true, cache.α)
    end

    for i in 1:(cache.maxiters)
        fxλp_norm = ϕ(λ₂)
        converged = fxλp_norm ≤ (1 + cache.η) * fx_norm - cache.σ₁ * λ₂^2 * du_norm^2
        converged && return (false, λ₂)
        λ₁, λ₂ = λ₂, cache.β * λ₂
    end

    return true, cache.α
end
