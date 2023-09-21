"""
    LineSearch(method = Static(), autodiff = AutoFiniteDiff(), alpha = true)

Wrapper over algorithms from
[LineSeaches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl/). Allows automatic
construction of the objective functions for the line search algorithms utilizing automatic
differentiation for fast Vector Jacobian Products.

### Arguments

  - `method`: the line search algorithm to use. Defaults to `Static()`, which means that the
    step size is fixed to the value of `alpha`.
  - `autodiff`: the automatic differentiation backend to use for the line search. Defaults to
    `AutoFiniteDiff()`, which means that finite differencing is used to compute the VJP.
    `AutoZygote()` will be faster in most cases, but it requires `Zygote.jl` to be manually
    installed and loaded
  - `alpha`: the initial step size to use. Defaults to `true` (which is equivalent to `1`).
"""
@concrete struct LineSearch
    method
    autodiff
    α
end

function LineSearch(; method = Static(), autodiff = AutoFiniteDiff(), alpha = true)
    return LineSearch(method, autodiff, alpha)
end

@concrete mutable struct LineSearchCache
    f
    ϕ
    dϕ
    ϕdϕ
    α
    ls
end

function LineSearchCache(ls::LineSearch, f, u::Number, p, _, ::Val{false})
    eval_f(u, du, α) = eval_f(u - α * du)
    eval_f(u) = f(u, p)

    ls.method isa Static && return LineSearchCache(eval_f, nothing, nothing, nothing,
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

    return LineSearchCache(eval_f, ϕ, dϕ, ϕdϕ, convert(eltype(u), ls.α), ls)
end

function LineSearchCache(ls::LineSearch, f, u, p, fu1, IIP::Val{iip}) where {iip}
    fu = iip ? fu1 : nothing
    u_ = _mutable_zero(u)

    function eval_f(u, du, α)
        @. u_ = u - α * du
        return eval_f(u_)
    end
    eval_f(u) = evaluate_f(f, u, p, IIP; fu)

    ls.method isa Static && return LineSearchCache(eval_f, nothing, nothing, nothing,
        convert(eltype(u), ls.α), ls)

    g₀ = _mutable_zero(u)

    autodiff = if iip && (ls.autodiff isa AutoZygote || ls.autodiff isa AutoSparseZygote)
        @warn "Attempting to use Zygote.jl for linesearch on an in-place problem. Falling back to finite differencing."
        AutoFiniteDiff()
    else
        ls.autodiff
    end

    function g!(u, fu)
        op = VecJac((args...) -> f(args..., p), u; autodiff)
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

    return LineSearchCache(eval_f, ϕ, dϕ, ϕdϕ, convert(eltype(u), ls.α), ls)
end

function perform_linesearch!(cache::LineSearchCache, u, du)
    cache.ls.method isa Static && return cache.α

    ϕ = cache.ϕ(u, du)
    dϕ = cache.dϕ(u, du)
    ϕdϕ = cache.ϕdϕ(u, du)

    ϕ₀, dϕ₀ = ϕdϕ(zero(eltype(u)))

    # This case is sometimes possible for large optimization problems
    dϕ₀ ≥ 0 && return cache.α

    return first(cache.ls.method(ϕ, cache.dϕ(u, du), cache.ϕdϕ(u, du), cache.α, ϕ₀, dϕ₀))
end
