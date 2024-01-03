# Not part of public API but helps reduce code duplication
import SimpleNonlinearsolve: __nlsolve_ad, __nlsolve_dual_soln

function SciMLBase.solve(prob::NonlinearProblem{<:Union{Number, <:AbstractArray},
            iip, <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}}},
        alg::Union{Nothing, AbstractNonlinearAlgorithm}, args...;
        kwargs...) where {T, V, P, iip}
    sol, partials = __nlsolve_ad(prob, alg, args...; kwargs...)
    dual_soln = __nlsolve_dual_soln(sol.u, partials, prob.p)
    return SciMLBase.build_solution(prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats,
        sol.original)
end

@concrete mutable struct NonlinearSolveForwardDiffCache
    cache
    prob
    alg
    p
    values_p
    partials_p
end

function SciMLBase.reinit!(cache::NonlinearSolveForwardDiffCache; p = cache.p,
        u0 = get_u(cache.cache), kwargs...)
    inner_cache = SciMLBase.reinit!(cache.cache; p = ForwardDiff.value(p),
        u0 = ForwardDiff.value(u0), kwargs...)
    cache.cache = inner_cache
    cache.p = p
    cache.values_p = ForwardDiff.value(p)
    cache.partials_p = ForwardDiff.partials(p)
    return cache
end

function SciMLBase.init(prob::NonlinearProblem{<:Union{Number, <:AbstractArray},
            iip, <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}}},
        alg::Union{Nothing, AbstractNonlinearAlgorithm}, args...;
        kwargs...) where {T, V, P, iip}
    p = ForwardDiff.value(prob.p)
    newprob = NonlinearProblem(prob.f, ForwardDiff.value(prob.u0), p; prob.kwargs...)
    cache = init(newprob, alg, args...; kwargs...)
    return NonlinearSolveForwardDiffCache(cache, newprob, alg, prob.p, p,
        ForwardDiff.partials(prob.p))
end

function SciMLBase.solve!(cache::NonlinearSolveForwardDiffCache)
    sol = solve!(cache.cache)
    prob = cache.prob

    uu = sol.u
    f_p = __nlsolve_∂f_∂p(prob, prob.f, uu, cache.values_p)
    f_x = __nlsolve_∂f_∂u(prob, prob.f, uu, cache.values_p)

    z_arr = -f_x \ f_p

    sumfun = ((z, p),) -> map(zᵢ -> zᵢ * ForwardDiff.partials(p), z)
    if cache.p isa Number
        partials = sumfun((z_arr, cache.p))
    else
        partials = sum(sumfun, zip(eachcol(z_arr), cache.p))
    end

    dual_soln = __nlsolve_dual_soln(sol.u, partials, cache.p)
    return SciMLBase.build_solution(prob, cache.alg, dual_soln, sol.resid; sol.retcode,
        sol.stats, sol.original)
end
