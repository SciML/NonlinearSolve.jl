"""
    HalleyDescent(; linsolve = nothing, precs = DEFAULT_PRECS)

Improve the NewtonDescent with higher-order terms. First compute the descent direction as ``J a = -fu``.
Then compute the hessian-vector-vector product and solve for the second-order correction term as ``J b = H a a``.
Finally, compute the descent direction as ``δu = a * a / (b / 2 - a)``.

See also [`NewtonDescent`](@ref).
"""
@kwdef @concrete struct HalleyDescent <: AbstractDescentAlgorithm
    linsolve = nothing
    precs = DEFAULT_PRECS
end

using TaylorDiff: derivative

function Base.show(io::IO, d::HalleyDescent)
    modifiers = String[]
    d.linsolve !== nothing && push!(modifiers, "linsolve = $(d.linsolve)")
    d.precs !== DEFAULT_PRECS && push!(modifiers, "precs = $(d.precs)")
    print(io, "HalleyDescent($(join(modifiers, ", ")))")
end

supports_line_search(::HalleyDescent) = true

@concrete mutable struct HalleyDescentCache{pre_inverted} <: AbstractDescentCache
    f
    p
    δu
    δus
    b
    fu
    lincache
    timer
end

@internal_caches HalleyDescentCache :lincache

function __internal_init(
        prob::NonlinearProblem, alg::HalleyDescent, J, fu, u; shared::Val{N} = Val(1),
        pre_inverted::Val{INV} = False, linsolve_kwargs = (;), abstol = nothing,
        reltol = nothing, timer = get_timer_output(), kwargs...) where {INV, N}
    @bb δu = similar(u)
    @bb b = similar(u)
    @bb fu = similar(fu)
    δus = N ≤ 1 ? nothing : map(2:N) do i
        @bb δu_ = similar(u)
    end
    INV && return HalleyDescentCache{true}(prob.f, prob.p, δu, δus, b, nothing, timer)
    lincache = LinearSolverCache(
        alg, alg.linsolve, J, _vec(fu), _vec(u); abstol, reltol, linsolve_kwargs...)
    return HalleyDescentCache{false}(prob.f, prob.p, δu, δus, b, fu, lincache, timer)
end

function __internal_solve!(cache::HalleyDescentCache{INV}, J, fu, u, idx::Val = Val(1);
        skip_solve::Bool = false, new_jacobian::Bool = true, kwargs...) where {INV}
    δu = get_du(cache, idx)
    skip_solve && return δu, true, (;)
    if INV
        @assert J!==nothing "`J` must be provided when `pre_inverted = Val(true)`."
        @bb δu = J × vec(fu)
    else
        @static_timeit cache.timer "linear solve 1" begin
            δu = cache.lincache(;
                A = J, b = _vec(fu), kwargs..., linu = _vec(δu), du = _vec(δu),
                reuse_A_if_factorization = !new_jacobian || (idx !== Val(1)))
            δu = _restructure(get_du(cache, idx), δu)
        end
    end
    b = cache.b
    # compute the hessian-vector-vector product
    hvvp = evaluate_hvvp(cache, cache.f, cache.p, u, δu)
    # second linear solve, reuse factorization if possible
    if INV
        @bb b = J × vec(hvvp)
    else
        @static_timeit cache.timer "linear solve 2" begin
            b = cache.lincache(; A = J, b = _vec(hvvp), kwargs..., linu = _vec(b),
                du = _vec(b), reuse_A_if_factorization = true)
            b = _restructure(cache.b, b)
        end
    end
    @bb @. δu = δu * δu / (b / 2 - δu)
    set_du!(cache, δu, idx)
    cache.b = b
    return δu, true, (;)
end

function evaluate_hvvp(
        cache::HalleyDescentCache, f::NonlinearFunction{iip}, p, u, δu) where {iip}
    if iip
        binary_f = (y, x) -> f(y, x, p)
        derivative(binary_f, cache.fu, u, δu, Val{3}())
    else
        unary_f = Base.Fix2(f, p)
        derivative(unary_f, u, δu, Val{3}())
    end
end
