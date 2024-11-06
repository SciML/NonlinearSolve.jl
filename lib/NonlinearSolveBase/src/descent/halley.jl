"""
    HalleyDescent(; linsolve = nothing)

Improve the NewtonDescent with higher-order terms. First compute the descent direction as ``J a = -fu``.
Then compute the hessian-vector-vector product and solve for the second-order correction term as ``J b = H a a``.
Finally, compute the descent direction as ``δu = a * a / (b / 2 - a)``.

Note that `import TaylorDiff` is required to use this descent algorithm.

See also [`NewtonDescent`](@ref).
"""
@kwdef @concrete struct HalleyDescent <: AbstractDescentDirection
    linsolve = nothing
end

supports_line_search(::HalleyDescent) = true

@concrete mutable struct HalleyDescentCache <: AbstractDescentCache
    f
    p
    δu
    δus
    b
    fu
    hvvp
    lincache
    timer
    preinverted_jacobian <: Union{Val{false}, Val{true}}
end

@internal_caches HalleyDescentCache :lincache

function InternalAPI.init(
        prob::NonlinearProblem, alg::HalleyDescent, J, fu, u; stats,
        shared = Val(1), pre_inverted::Val = Val(false),
        linsolve_kwargs = (;), abstol = nothing, reltol = nothing,
        timer = get_timer_output(), kwargs...)
    @bb δu = similar(u)
    @bb b = similar(u)
    @bb fu = similar(fu)
    @bb hvvp = similar(fu)
    δus = Utils.unwrap_val(shared) ≤ 1 ? nothing : map(2:Utils.unwrap_val(shared)) do i
        @bb δu_ = similar(u)
    end
    lincache = Utils.unwrap_val(pre_inverted) ? nothing :
               construct_linear_solver(
        alg, alg.linsolve, J, Utils.safe_vec(fu), Utils.safe_vec(u);
        stats, abstol, reltol, linsolve_kwargs...
    )
    return HalleyDescentCache(
        prob.f, prob.p, δu, δus, b, fu, hvvp, lincache, timer, pre_inverted)
end

function InternalAPI.solve!(
        cache::HalleyDescentCache, J, fu, u, idx::Val = Val(1);
        skip_solve::Bool = false, new_jacobian::Bool = true, kwargs...)
    δu = SciMLBase.get_du(cache, idx)
    skip_solve && return DescentResult(; δu)
    if preinverted_jacobian(cache)
        @assert J!==nothing "`J` must be provided when `pre_inverted = Val(true)`."
        @bb δu = J × vec(fu)
    else
        @static_timeit cache.timer "linear solve 1" begin
            linres = cache.lincache(;
                A = J, b = Utils.safe_vec(fu),
                kwargs..., linu = Utils.safe_vec(δu),
                reuse_A_if_factorization = !new_jacobian || (idx !== Val(1)))
            δu = Utils.restructure(SciMLBase.get_du(cache, idx), linres.u)
            if !linres.success
                set_du!(cache, δu, idx)
                return DescentResult(; δu, success = false, linsolve_success = false)
            end
        end
    end
    b = cache.b
    # compute the hessian-vector-vector product
    hvvp = evaluate_hvvp(cache.hvvp, cache, cache.f, cache.p, u, δu)
    # second linear solve, reuse factorization if possible
    if preinverted_jacobian(cache)
        @bb b = J × vec(hvvp)
    else
        @static_timeit cache.timer "linear solve 2" begin
            linres = cache.lincache(;
                A = J, b = Utils.safe_vec(hvvp),
                kwargs..., linu = Utils.safe_vec(b),
                reuse_A_if_factorization = true)
            b = Utils.restructure(cache.b, linres.u)
            if !linres.success
                set_du!(cache, δu, idx)
                return DescentResult(; δu, success = false, linsolve_success = false)
            end
        end
    end
    @bb @. δu = δu * δu / (b / 2 - δu)
    set_du!(cache, δu, idx)
    cache.b = b
    return DescentResult(; δu)
end

evaluate_hvvp(hvvp, cache, f, p, u, δu) = error("not implemented. please import TaylorDiff")
