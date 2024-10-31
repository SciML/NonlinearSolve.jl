const DualNonlinearProblem = NonlinearProblem{
    <:Union{Number, <:AbstractArray}, iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}}
} where {iip, T, V, P}
const DualNonlinearLeastSquaresProblem = NonlinearLeastSquaresProblem{
    <:Union{Number, <:AbstractArray}, iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}}
} where {iip, T, V, P}
const DualAbstractNonlinearProblem = Union{
    DualNonlinearProblem, DualNonlinearLeastSquaresProblem
}

for algType in ALL_SOLVER_TYPES
    @eval function SciMLBase.__solve(
            prob::DualAbstractNonlinearProblem, alg::$(algType), args...; kwargs...
    )
        sol, partials = NonlinearSolveBase.nonlinearsolve_forwarddiff_solve(
            prob, alg, args...; kwargs...
        )
        dual_soln = NonlinearSolveBase.nonlinearsolve_dual_solution(sol.u, partials, prob.p)
        return SciMLBase.build_solution(
            prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original
        )
    end
end

@concrete mutable struct NonlinearSolveForwardDiffCache <: AbstractNonlinearSolveCache
    cache
    prob
    alg
    p
    values_p
    partials_p
end

function InternalAPI.reinit!(
        cache::NonlinearSolveForwardDiffCache, args...;
        p = cache.p, u0 = NonlinearSolveBase.get_u(cache.cache), kwargs...
)
    inner_cache = InternalAPI.reinit!(
        cache.cache; p = nodual_value(p), u0 = nodual_value(u0), kwargs...
    )
    cache.cache = inner_cache
    cache.p = p
    cache.values_p = nodual_value(p)
    cache.partials_p = ForwardDiff.partials(p)
    return cache
end

for algType in ALL_SOLVER_TYPES
    # XXX: Extend to DualNonlinearLeastSquaresProblem
    @eval function SciMLBase.__init(
            prob::DualNonlinearProblem, alg::$(algType), args...; kwargs...
    )
        p = nodual_value(prob.p)
        newprob = SciMLBase.remake(prob; u0 = nodual_value(prob.u0), p)
        cache = init(newprob, alg, args...; kwargs...)
        return NonlinearSolveForwardDiffCache(
            cache, newprob, alg, prob.p, p, ForwardDiff.partials(prob.p)
        )
    end
end

function CommonSolve.solve!(cache::NonlinearSolveForwardDiffCache)
    sol = solve!(cache.cache)
    prob = cache.prob

    uu = sol.u
    Jₚ = NonlinearSolveBase.nonlinearsolve_∂f_∂p(prob, prob.f, uu, cache.values_p)
    Jᵤ = NonlinearSolveBase.nonlinearsolve_∂f_∂u(prob, prob.f, uu, cache.values_p)

    z_arr = -Jᵤ \ Jₚ

    sumfun = ((z, p),) -> map(zᵢ -> zᵢ * ForwardDiff.partials(p), z)
    if cache.p isa Number
        partials = sumfun((z_arr, cache.p))
    else
        partials = sum(sumfun, zip(eachcol(z_arr), cache.p))
    end

    dual_soln = NonlinearSolveBase.nonlinearsolve_dual_solution(sol.u, partials, cache.p)
    return SciMLBase.build_solution(
        prob, cache.alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original
    )
end

nodual_value(x) = x
nodual_value(x::Dual) = ForwardDiff.value(x)
nodual_value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)

"""
    pickchunksize(x) = pickchunksize(length(x))
    pickchunksize(x::Int)

Determine the chunk size for ForwardDiff and PolyesterForwardDiff based on the input length.
"""
@inline pickchunksize(x) = pickchunksize(length(x))
@inline pickchunksize(x::Int) = ForwardDiff.pickchunksize(x)
