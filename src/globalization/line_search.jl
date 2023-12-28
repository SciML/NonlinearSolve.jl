"""
    NoLineSearch <: AbstractNonlinearSolveLineSearchAlgorithm

Don't perform a line search. Just return the initial step length of `1`.
"""
struct NoLineSearch <: AbstractNonlinearSolveLineSearchAlgorithm end

@concrete struct NoLineSearchCache
    α
end

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::NoLineSearch, f::F, fu, u,
        p, args...; kwargs...) where {F}
    return NoLineSearchCache(promote_type(eltype(fu), eltype(u))(true))
end

SciMLBase.solve!(cache::NoLineSearchCache, u, du) = cache.α

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
  - `alpha`: the initial step size to use. Defaults to `true` (which is equivalent to `1`).
"""
@concrete struct LineSearchesJL <: AbstractNonlinearSolveLineSearchAlgorithm
    method
    initial_alpha
    autodiff
end

LineSearchesJL(method; kwargs...) = LineSearchesJL(; method, kwargs...)
function LineSearchesJL(; method = LineSearches.Static(), autodiff = nothing, α = true)
    return LineSearchesJL(method, α, autodiff)
end

Base.@deprecate_binding LineSearch LineSearchesJL true

# Wrapper over LineSearches.jl algorithms
@concrete mutable struct LineSearchesJLCache
    ϕ
    dϕ
    ϕdϕ
    method
    alpha
end

get_fu(cache::LineSearchesJLCache) = cache.fu
set_fu!(cache::LineSearchesJLCache, fu) = (cache.fu = fu)

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::LineSearchesJL, f::F, fu, u,
        p, args...; internalnorm::IN = DEFAULT_NORM, kwargs...) where {F, IN}
    T = promote_type(eltype(fu), eltype(u))
    if u isa Number
        grad_op = @closure (u, fu) -> begin
            last(__value_derivative(Base.Fix2(f, p), u)) * fu
        end
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
            grad_op = @closure (u, fu) -> begin
                # op = VecJac(SciMLBase.JacobianWrapper(f, p), u; fu = fu1, autodiff)
                # if iip
                #     mul!(g₀, op, fu)
                #     return g₀
                # else
                #     return op * fu
                # end
                error("Not Implemented Yet!")
            end
        end
    end

    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)

    ϕ = @closure (u, du, α) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(prob, fu_cache, u_cache, p)
        return @fastmath internalnorm(fu_cache)^2 / 2
    end

    dϕ = @closure (u, du, α) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(prob, fu_cache, u_cache, p)
        g₀ = grad_op(u_cache, fu_cache)
        return dot(g₀, du)
    end

    ϕdϕ = @closure (u, du, α) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(prob, fu_cache, u_cache, p)
        g₀ = grad_op(u_cache, fu_cache)
        obj = @fastmath internalnorm(fu_cache)^2 / 2
        return obj, dot(g₀, du)
    end

    return LineSearchesJLCache(ϕ, dϕ, ϕdϕ, alg.method, T(alg.initial_alpha))
end

function SciMLBase.solve!(cache::LineSearchesJLCache, u, du)
    ϕ = @closure α -> cache.ϕ(u, du, α)
    dϕ = @closure α -> cache.dϕ(u, du, α)
    ϕdϕ = @closure α -> cache.ϕdϕ(u, du, α)

    ϕ₀, dϕ₀ = ϕdϕ(zero(eltype(u)))

    # Here we should be resetting the search direction for some algorithms especially
    # if we start mixing in jacobian reuse and such
    dϕ₀ ≥ 0 && return one(eltype(u))

    # We can technically reduce 1 axpy by reusing the returned value from cache.method
    # but it's not worth the extra complexity
    cache.alpha = first(cache.method(ϕ, dϕ, ϕdϕ, cache.alpha, ϕ₀, dϕ₀))
    return cache.alpha
end

# """
#     LiFukushimaLineSearch(; lambda_0 = 1.0, beta = 0.5, sigma_1 = 0.001,
#         eta = 0.1, nan_max_iter = 5, maxiters = 50)

# A derivative-free line search and global convergence of Broyden-like method for nonlinear
# equations by Dong-Hui Li & Masao Fukushima. For more details see
# https://doi.org/10.1080/10556780008805782
# """
# struct LiFukushimaLineSearch{T} <: AbstractNonlinearSolveLineSearchAlgorithm
#     λ₀::T
#     β::T
#     σ₁::T
#     σ₂::T
#     η::T
#     ρ::T
#     nan_max_iter::Int
#     maxiters::Int
# end

# function LiFukushimaLineSearch(; lambda_0 = 1.0, beta = 0.1, sigma_1 = 0.001,
#         sigma_2 = 0.001, eta = 0.1, rho = 0.9, nan_max_iter = 5, maxiters = 50)
#     T = promote_type(typeof(lambda_0), typeof(beta), typeof(sigma_1), typeof(eta),
#         typeof(rho), typeof(sigma_2))
#     return LiFukushimaLineSearch{T}(lambda_0, beta, sigma_1, sigma_2, eta, rho,
#         nan_max_iter, maxiters)
# end

# @concrete mutable struct LiFukushimaLineSearchCache{iip}
#     f
#     p
#     u_cache
#     fu_cache
#     alg
#     α
# end

# function init_linesearch_cache(alg::LiFukushimaLineSearch, ls::LineSearch, f::F, _u, p, _fu,
#         ::Val{iip}) where {iip, F}
#     fu = iip ? deepcopy(_fu) : nothing
#     u = iip ? deepcopy(_u) : nothing
#     return LiFukushimaLineSearchCache{iip}(f, p, u, fu, alg, ls.α)
# end

# function perform_linesearch!(cache::LiFukushimaLineSearchCache{iip}, u, du) where {iip}
#     (; β, σ₁, σ₂, η, λ₀, ρ, nan_max_iter, maxiters) = cache.alg
#     λ₂ = λ₀
#     λ₁ = λ₂

#     if iip
#         cache.f(cache.fu_cache, u, cache.p)
#         fx_norm = norm(cache.fu_cache, 2)
#     else
#         fx_norm = norm(cache.f(u, cache.p), 2)
#     end

#     # Non-Blocking exit if the norm is NaN or Inf
#     !isfinite(fx_norm) && return cache.α

#     # Early Terminate based on Eq. 2.7
#     if iip
#         cache.u_cache .= u .- du
#         cache.f(cache.fu_cache, cache.u_cache, cache.p)
#         fxλ_norm = norm(cache.fu_cache, 2)
#     else
#         fxλ_norm = norm(cache.f(u .- du, cache.p), 2)
#     end

#     fxλ_norm ≤ ρ * fx_norm - σ₂ * norm(du, 2)^2 && return cache.α

#     if iip
#         cache.u_cache .= u .- λ₂ .* du
#         cache.f(cache.fu_cache, cache.u_cache, cache.p)
#         fxλp_norm = norm(cache.fu_cache, 2)
#     else
#         fxλp_norm = norm(cache.f(u .- λ₂ .* du, cache.p), 2)
#     end

#     if !isfinite(fxλp_norm)
#         # Backtrack a finite number of steps
#         nan_converged = false
#         for _ in 1:nan_max_iter
#             λ₁, λ₂ = λ₂, β * λ₂

#             if iip
#                 cache.u_cache .= u .+ λ₂ .* du
#                 cache.f(cache.fu_cache, cache.u_cache, cache.p)
#                 fxλp_norm = norm(cache.fu_cache, 2)
#             else
#                 fxλp_norm = norm(cache.f(u .+ λ₂ .* du, cache.p), 2)
#             end

#             nan_converged = isfinite(fxλp_norm)
#             nan_converged && break
#         end

#         # Non-Blocking exit if the norm is still NaN or Inf
#         !nan_converged && return cache.α
#     end

#     for _ in 1:maxiters
#         if iip
#             cache.u_cache .= u .- λ₂ .* du
#             cache.f(cache.fu_cache, cache.u_cache, cache.p)
#             fxλp_norm = norm(cache.fu_cache, 2)
#         else
#             fxλp_norm = norm(cache.f(u .- λ₂ .* du, cache.p), 2)
#         end

#         converged = fxλp_norm ≤ (1 + η) * fx_norm - σ₁ * λ₂^2 * norm(du, 2)^2

#         converged && break
#         λ₁, λ₂ = λ₂, β * λ₂
#     end

#     return λ₂
# end
