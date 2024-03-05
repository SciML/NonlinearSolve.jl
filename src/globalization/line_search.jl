"""
    NoLineSearch <: AbstractNonlinearSolveLineSearchAlgorithm

Don't perform a line search. Just return the initial step length of `1`.
"""
struct NoLineSearch <: AbstractNonlinearSolveLineSearchAlgorithm end

@concrete mutable struct NoLineSearchCache <: AbstractNonlinearSolveLineSearchCache
    α
end

function __internal_init(prob::AbstractNonlinearProblem, alg::NoLineSearch,
        f::F, fu, u, p, args...; kwargs...) where {F}
    return NoLineSearchCache(promote_type(eltype(fu), eltype(u))(true))
end

reinit_cache!(cache::NoLineSearchCache, args...; p = cache.p, kwargs...) = nothing

__internal_solve!(cache::NoLineSearchCache, u, du) = false, cache.α

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
  - `autodiff`: the automatic differentiation backend to use for the line search. Using a
    reverse mode automatic differentiation backend if recommended.
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
    if method isa LineSearchesJL  # Prevent breaking old code
        return LineSearchesJL(method.method, α, autodiff)
    end

    if method isa AbstractNonlinearSolveLineSearchAlgorithm
        Base.depwarn("Passing a native NonlinearSolve line search algorithm to \
                      `LineSearchesJL` or `LineSearch` is deprecated. Pass the method \
                      directly instead.",
            :LineSearchesJL)
        return method
    end
    return LineSearchesJL(method, α, autodiff)
end

Base.@deprecate_binding LineSearch LineSearchesJL true

Static(args...; kwargs...) = LineSearchesJL(LineSearches.Static(args...; kwargs...))
HagerZhang(args...; kwargs...) = LineSearchesJL(LineSearches.HagerZhang(args...; kwargs...))
function MoreThuente(args...; kwargs...)
    return LineSearchesJL(LineSearches.MoreThuente(args...; kwargs...))
end
function BackTracking(args...; kwargs...)
    return LineSearchesJL(LineSearches.BackTracking(args...; kwargs...))
end
function StrongWolfe(args...; kwargs...)
    return LineSearchesJL(LineSearches.StrongWolfe(args...; kwargs...))
end

# Wrapper over LineSearches.jl algorithms
@concrete mutable struct LineSearchesJLCache <: AbstractNonlinearSolveLineSearchCache
    f
    p
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

function __internal_init(
        prob::AbstractNonlinearProblem, alg::LineSearchesJL, f::F, fu, u, p,
        args...; internalnorm::IN = DEFAULT_NORM, kwargs...) where {F, IN}
    T = promote_type(eltype(fu), eltype(u))
    if u isa Number
        grad_op = @closure (u, fu, p) -> last(__value_derivative(Base.Fix2(f, p), u)) * fu
    else
        if SciMLBase.has_jvp(f)
            if isinplace(prob)
                g_cache = similar(u)
                grad_op = @closure (u, fu, p) -> f.vjp(g_cache, fu, u, p)
            else
                grad_op = @closure (u, fu, p) -> f.vjp(fu, u, p)
            end
        else
            autodiff = get_concrete_reverse_ad(
                alg.autodiff, prob; check_forward_mode = true)
            vjp_op = VecJacOperator(prob, fu, u; autodiff)
            if isinplace(prob)
                g_cache = similar(u)
                grad_op = @closure (u, fu, p) -> vjp_op(g_cache, fu, u, p)
            else
                grad_op = @closure (u, fu, p) -> vjp_op(fu, u, p)
            end
        end
    end

    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    nf = Base.RefValue(0)

    ϕ = @closure (f, p, u, du, α, u_cache, fu_cache) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(f, fu_cache, u_cache, p)
        nf[] += 1
        return @fastmath internalnorm(fu_cache)^2 / 2
    end

    dϕ = @closure (f, p, u, du, α, u_cache, fu_cache, grad_op) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(f, fu_cache, u_cache, p)
        nf[] += 1
        g₀ = grad_op(u_cache, fu_cache, p)
        return dot(g₀, du)
    end

    ϕdϕ = @closure (f, p, u, du, α, u_cache, fu_cache, grad_op) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(f, fu_cache, u_cache, p)
        nf[] += 1
        g₀ = grad_op(u_cache, fu_cache, p)
        obj = @fastmath internalnorm(fu_cache)^2 / 2
        return obj, dot(g₀, du)
    end

    return LineSearchesJLCache(
        f, p, ϕ, dϕ, ϕdϕ, alg.method, T(alg.initial_alpha), grad_op, u_cache, fu_cache, nf)
end

function __internal_solve!(cache::LineSearchesJLCache, u, du; kwargs...)
    ϕ = @closure α -> cache.ϕ(cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache)
    dϕ = @closure α -> cache.dϕ(
        cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache, cache.grad_op)
    ϕdϕ = @closure α -> cache.ϕdϕ(
        cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache, cache.grad_op)

    ϕ₀, dϕ₀ = ϕdϕ(zero(eltype(u)))

    # Here we should be resetting the search direction for some algorithms especially
    # if we start mixing in jacobian reuse and such
    dϕ₀ ≥ 0 && return (true, one(eltype(u)))

    # We can technically reduce 1 axpy by reusing the returned value from cache.method
    # but it's not worth the extra complexity
    cache.alpha = first(cache.method(ϕ, dϕ, ϕdϕ, cache.alpha, ϕ₀, dϕ₀))
    return (false, cache.alpha)
end

"""
    RobustNonMonotoneLineSearch(; gamma = 1 // 10000, sigma_0 = 1, M::Int = 10,
        tau_min = 1 // 10, tau_max = 1 // 2, n_exp::Int = 2, maxiters::Int = 100,
        η_strategy = (fn₁, n, uₙ, fₙ) -> fn₁ / n^2)

Robust NonMonotone Line Search is a derivative free line search method from DF Sane
[la2006spectral](@cite).

### Keyword Arguments

  - `M`: The monotonicity of the algorithm is determined by a this positive integer.
    A value of 1 for `M` would result in strict monotonicity in the decrease of the L2-norm
    of the function `f`. However, higher values allow for more flexibility in this reduction.
    Despite this, the algorithm still ensures global convergence through the use of a
    non-monotone line-search algorithm that adheres to the Grippo-Lampariello-Lucidi
    condition. Values in the range of 5 to 20 are usually sufficient, but some cases may
    call for a higher value of `M`. The default setting is 10.
  - `gamma`: a parameter that influences if a proposed step will be accepted. Higher value
    of `gamma` will make the algorithm more restrictive in accepting steps. Defaults to
    `1e-4`.
  - `tau_min`: if a step is rejected the new step size will get multiplied by factor, and
    this parameter is the minimum value of that factor. Defaults to `0.1`.
  - `tau_max`: if a step is rejected the new step size will get multiplied by factor, and
    this parameter is the maximum value of that factor. Defaults to `0.5`.
  - `n_exp`: the exponent of the loss, i.e. ``f_n=||F(x_n)||^{n\\_exp}``. The paper uses
    `n_exp ∈ {1, 2}`. Defaults to `2`.
  - `η_strategy`:  function to determine the parameter `η`, which enables growth
    of ``||f_n||^2``. Called as `η = η_strategy(fn_1, n, x_n, f_n)` with `fn_1` initialized
    as ``fn_1=||f(x_1)||^{n\\_exp}``, `n` is the iteration number, `x_n` is the current
    `x`-value and `f_n` the current residual. Should satisfy ``η > 0`` and ``∑ₖ ηₖ < ∞``.
    Defaults to ``fn_1 / n^2``.
  - `maxiters`: the maximum number of iterations allowed for the inner loop of the
    algorithm. Defaults to `100`.
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
    f
    p
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

function __internal_init(
        prob::AbstractNonlinearProblem, alg::RobustNonMonotoneLineSearch, f::F, fu,
        u, p, args...; internalnorm::IN = DEFAULT_NORM, kwargs...) where {F, IN}
    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    T = promote_type(eltype(fu), eltype(u))

    nf = Base.RefValue(0)
    ϕ = @closure (f, p, u, du, α, u_cache, fu_cache) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(f, fu_cache, u_cache, p)
        nf[] += 1
        return internalnorm(fu_cache)^alg.n_exp
    end

    fn₁ = internalnorm(fu)^alg.n_exp
    η_strategy = @closure (n, xₙ, fₙ) -> alg.η_strategy(fn₁, n, xₙ, fₙ)

    return RobustNonMonotoneLineSearchCache(
        f, p, ϕ, u_cache, fu_cache, internalnorm, alg.maxiters,
        fill(fn₁, alg.M), T(alg.gamma), T(alg.sigma_1), alg.M,
        T(alg.tau_min), T(alg.tau_max), 0, η_strategy, alg.n_exp, nf)
end

function __internal_solve!(cache::RobustNonMonotoneLineSearchCache, u, du; kwargs...)
    T = promote_type(eltype(u), eltype(du))
    ϕ = @closure α -> cache.ϕ(cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache)
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
equations [li2000derivative](@cite).
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

function __internal_init(
        prob::AbstractNonlinearProblem, alg::LiFukushimaLineSearch, f::F, fu, u,
        p, args...; internalnorm::IN = DEFAULT_NORM, kwargs...) where {F, IN}
    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    T = promote_type(eltype(fu), eltype(u))

    nf = Base.RefValue(0)
    ϕ = @closure (f, p, u, du, α, u_cache, fu_cache) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(f, fu_cache, u_cache, p)
        nf[] += 1
        return internalnorm(fu_cache)
    end

    return LiFukushimaLineSearchCache(
        ϕ, f, p, internalnorm, u_cache, fu_cache, T(alg.lambda_0),
        T(alg.beta), T(alg.sigma_1), T(alg.sigma_2), T(alg.eta),
        T(alg.rho), T(true), alg.nan_max_iter, alg.maxiters, nf)
end

function __internal_solve!(cache::LiFukushimaLineSearchCache, u, du; kwargs...)
    T = promote_type(eltype(u), eltype(du))
    ϕ = @closure α -> cache.ϕ(cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache)

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
