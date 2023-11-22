"""
    LineSearch(method = nothing, autodiff = nothing, alpha = true)

Wrapper over algorithms from
[LineSeaches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl/). Allows automatic
construction of the objective functions for the line search algorithms utilizing automatic
differentiation for fast Vector Jacobian Products.

### Arguments

  - `method`: the line search algorithm to use. Defaults to `nothing`, which means that the
    step size is fixed to the value of `alpha`.
  - `autodiff`: the automatic differentiation backend to use for the line search. Defaults to
    `AutoFiniteDiff()`, which means that finite differencing is used to compute the VJP.
    `AutoZygote()` will be faster in most cases, but it requires `Zygote.jl` to be manually
    installed and loaded.
  - `alpha`: the initial step size to use. Defaults to `true` (which is equivalent to `1`).
"""
@concrete struct LineSearch
    method
    autodiff
    α
end

function LineSearch(; method = nothing, autodiff = nothing, alpha = true)
    return LineSearch(method, autodiff, alpha)
end

@inline function init_linesearch_cache(ls::LineSearch, f::F, u, p, fu, iip) where {F}
    return init_linesearch_cache(ls.method, ls, f, u, p, fu, iip)
end

@concrete struct NoLineSearchCache
    α
end

function init_linesearch_cache(::Nothing, ls::LineSearch, f::F, u, p, fu, iip) where {F}
    return NoLineSearchCache(convert(eltype(u), ls.α))
end

perform_linesearch!(cache::NoLineSearchCache, u, du) = cache.α

# LineSearches.jl doesn't have a supertype so default to that
function init_linesearch_cache(_, ls::LineSearch, f::F, u, p, fu, iip) where {F}
    return LineSearchesJLCache(ls, f, u, p, fu, iip)
end

# FIXME: The closures lead to too many unnecessary runtime dispatches which leads to the
#        massive increase in precompilation times.
# Wrapper over LineSearches.jl algorithms
@concrete mutable struct LineSearchesJLCache
    f
    ϕ
    dϕ
    ϕdϕ
    α
    ls
end

function LineSearchesJLCache(ls::LineSearch, f::F, u::Number, p, _, ::Val{false}) where {F}
    eval_f(u, du, α) = eval_f(u - α * du)
    eval_f(u) = f(u, p)

    ls.method isa Static && return LineSearchesJLCache(eval_f, nothing, nothing, nothing,
        convert(typeof(u), ls.α), ls)

    g(u, fu) = last(value_derivative(Base.Fix2(f, p), u)) * fu

    function ϕ(u, du)
        function ϕ_internal(α)
            u_ = u - α * du
            _fu = eval_f(u_)
            return dot(_fu, _fu) / 2
        end
        return ϕ_internal
    end

    function dϕ(u, du)
        function dϕ_internal(α)
            u_ = u - α * du
            _fu = eval_f(u_)
            g₀ = g(u_, _fu)
            return dot(g₀, -du)
        end
        return dϕ_internal
    end

    function ϕdϕ(u, du)
        function ϕdϕ_internal(α)
            u_ = u - α * du
            _fu = eval_f(u_)
            g₀ = g(u_, _fu)
            return dot(_fu, _fu) / 2, dot(g₀, -du)
        end
        return ϕdϕ_internal
    end

    return LineSearchesJLCache(eval_f, ϕ, dϕ, ϕdϕ, convert(eltype(u), ls.α), ls)
end

function LineSearchesJLCache(ls::LineSearch, f::F, u, p, fu1, IIP::Val{iip}) where {iip, F}
    fu = iip ? deepcopy(fu1) : nothing
    u_ = _mutable_zero(u)

    function eval_f(u, du, α)
        @. u_ = u - α * du
        return eval_f(u_)
    end
    eval_f(u) = evaluate_f(f, u, p, IIP; fu)

    ls.method isa Static && return LineSearchesJLCache(eval_f, nothing, nothing, nothing,
        convert(eltype(u), ls.α), ls)

    g₀ = _mutable_zero(u)

    autodiff = if ls.autodiff === nothing
        if !iip && is_extension_loaded(Val{:Zygote}())
            AutoZygote()
        else
            AutoFiniteDiff()
        end
    else
        if iip && (ls.autodiff isa AutoZygote || ls.autodiff isa AutoSparseZygote)
            @warn "Attempting to use Zygote.jl for linesearch on an in-place problem. \
                Falling back to finite differencing."
            AutoFiniteDiff()
        else
            ls.autodiff
        end
    end

    function g!(u, fu)
        if f.jvp !== nothing
            @warn "Currently we don't make use of user provided `jvp` in linesearch. This \
                   is planned to be fixed in the near future." maxlog=1
        end
        op = VecJac(SciMLBase.JacobianWrapper(f, p), u; fu = fu1, autodiff)
        if iip
            mul!(g₀, op, fu)
            return g₀
        else
            return op * fu
        end
    end

    function ϕ(u, du)
        function ϕ_internal(α)
            @. u_ = u - α * du
            _fu = eval_f(u_)
            return dot(_fu, _fu) / 2
        end
        return ϕ_internal
    end

    function dϕ(u, du)
        function dϕ_internal(α)
            @. u_ = u - α * du
            _fu = eval_f(u_)
            g₀ = g!(u_, _fu)
            return dot(g₀, -du)
        end
        return dϕ_internal
    end

    function ϕdϕ(u, du)
        function ϕdϕ_internal(α)
            @. u_ = u - α * du
            _fu = eval_f(u_)
            g₀ = g!(u_, _fu)
            return dot(_fu, _fu) / 2, dot(g₀, -du)
        end
        return ϕdϕ_internal
    end

    return LineSearchesJLCache(eval_f, ϕ, dϕ, ϕdϕ, convert(eltype(u), ls.α), ls)
end

function perform_linesearch!(cache::LineSearchesJLCache, u, du)
    cache.ls.method isa Static && return cache.α

    ϕ = cache.ϕ(u, du)
    dϕ = cache.dϕ(u, du)
    ϕdϕ = cache.ϕdϕ(u, du)

    ϕ₀, dϕ₀ = ϕdϕ(zero(eltype(u)))

    return first(cache.ls.method(ϕ, dϕ, ϕdϕ, cache.α, ϕ₀, dϕ₀))
end

"""
    LiFukushimaLineSearch(; lambda_0 = 1.0, beta = 0.5, sigma_1 = 0.001,
        eta = 0.1, nan_max_iter = 5, maxiters = 50)

A derivative-free line search and global convergence of Broyden-like method for nonlinear
equations by Dong-Hui Li & Masao Fukushima. For more details see
https://doi.org/10.1080/10556780008805782
"""
struct LiFukushimaLineSearch{T} <: AbstractNonlinearSolveLineSearchAlgorithm
    λ₀::T
    β::T
    σ₁::T
    σ₂::T
    η::T
    ρ::T
    nan_max_iter::Int
    maxiters::Int
end

function LiFukushimaLineSearch(; lambda_0 = 1.0, beta = 0.1, sigma_1 = 0.001,
        sigma_2 = 0.001, eta = 0.1, rho = 0.9, nan_max_iter = 5, maxiters = 50)
    T = promote_type(typeof(lambda_0), typeof(beta), typeof(sigma_1), typeof(eta),
        typeof(rho), typeof(sigma_2))
    return LiFukushimaLineSearch{T}(lambda_0, beta, sigma_1, sigma_2, eta, rho,
        nan_max_iter, maxiters)
end

@concrete mutable struct LiFukushimaLineSearchCache{iip}
    f
    p
    u_cache
    fu_cache
    alg
    α
end

function init_linesearch_cache(alg::LiFukushimaLineSearch, ls::LineSearch, f::F, _u, p, _fu,
        ::Val{iip}) where {iip, F}
    fu = iip ? deepcopy(_fu) : nothing
    u = iip ? deepcopy(_u) : nothing
    return LiFukushimaLineSearchCache{iip}(f, p, u, fu, alg, ls.α)
end

function perform_linesearch!(cache::LiFukushimaLineSearchCache{iip}, u, du) where {iip}
    (; β, σ₁, σ₂, η, λ₀, ρ, nan_max_iter, maxiters) = cache.alg
    λ₂ = λ₀
    λ₁ = λ₂

    if iip
        cache.f(cache.fu_cache, u, cache.p)
        fx_norm = norm(cache.fu_cache, 2)
    else
        fx_norm = norm(cache.f(u, cache.p), 2)
    end

    # Non-Blocking exit if the norm is NaN or Inf
    !isfinite(fx_norm) && return cache.α

    # Early Terminate based on Eq. 2.7
    if iip
        cache.u_cache .= u .- du
        cache.f(cache.fu_cache, cache.u_cache, cache.p)
        fxλ_norm = norm(cache.fu_cache, 2)
    else
        fxλ_norm = norm(cache.f(u .- du, cache.p), 2)
    end

    fxλ_norm ≤ ρ * fx_norm - σ₂ * norm(du, 2)^2 && return cache.α

    if iip
        cache.u_cache .= u .- λ₂ .* du
        cache.f(cache.fu_cache, cache.u_cache, cache.p)
        fxλp_norm = norm(cache.fu_cache, 2)
    else
        fxλp_norm = norm(cache.f(u .- λ₂ .* du, cache.p), 2)
    end

    if !isfinite(fxλp_norm)
        # Backtrack a finite number of steps
        nan_converged = false
        for _ in 1:nan_max_iter
            λ₁, λ₂ = λ₂, β * λ₂

            if iip
                cache.u_cache .= u .+ λ₂ .* du
                cache.f(cache.fu_cache, cache.u_cache, cache.p)
                fxλp_norm = norm(cache.fu_cache, 2)
            else
                fxλp_norm = norm(cache.f(u .+ λ₂ .* du, cache.p), 2)
            end

            nan_converged = isfinite(fxλp_norm)
            nan_converged && break
        end

        # Non-Blocking exit if the norm is still NaN or Inf
        !nan_converged && return cache.α
    end

    for _ in 1:maxiters
        if iip
            cache.u_cache .= u .- λ₂ .* du
            cache.f(cache.fu_cache, cache.u_cache, cache.p)
            fxλp_norm = norm(cache.fu_cache, 2)
        else
            fxλp_norm = norm(cache.f(u .- λ₂ .* du, cache.p), 2)
        end

        converged = fxλp_norm ≤ (1 + η) * fx_norm - σ₁ * λ₂^2 * norm(du, 2)^2

        converged && break
        λ₁, λ₂ = λ₂, β * λ₂
    end

    return λ₂
end
