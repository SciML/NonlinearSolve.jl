const DualNonlinearProblem = NonlinearProblem{<:Union{Number, <:AbstractArray}, iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}}} where {iip, T, V, P}
const DualNonlinearLeastSquaresProblem = NonlinearLeastSquaresProblem{
    <:Union{Number, <:AbstractArray}, iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}}} where {iip, T, V, P}
const DualAbstractNonlinearProblem = Union{
    DualNonlinearProblem, DualNonlinearLeastSquaresProblem}

for algType in (
    Nothing, AbstractNonlinearSolveAlgorithm, GeneralizedDFSane,
    GeneralizedFirstOrderAlgorithm, ApproximateJacobianSolveAlgorithm,
    LeastSquaresOptimJL, FastLevenbergMarquardtJL, CMINPACK, NLsolveJL, NLSolversJL,
    SpeedMappingJL, FixedPointAccelerationJL, SIAMFANLEquationsJL,
    NonlinearSolvePolyAlgorithm{:NLLS, <:Any}, NonlinearSolvePolyAlgorithm{:NLS, <:Any}
)
    @eval function SciMLBase.__solve(
            prob::DualNonlinearProblem, alg::$(algType), args...; kwargs...)
        sol, partials = nonlinearsolve_forwarddiff_solve(prob, alg, args...; kwargs...)
        dual_soln = nonlinearsolve_dual_solution(sol.u, partials, prob.p)
        return SciMLBase.build_solution(
            prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original)
    end
end

@concrete mutable struct NonlinearSolveForwardDiffCache
    cache
    prob
    alg
    p
    values_p
    partials_p
end

@internal_caches NonlinearSolveForwardDiffCache :cache

function reinit_cache!(cache::NonlinearSolveForwardDiffCache;
        p = cache.p, u0 = get_u(cache.cache), kwargs...)
    inner_cache = reinit_cache!(cache.cache; p = __value(p), u0 = __value(u0), kwargs...)
    cache.cache = inner_cache
    cache.p = p
    cache.values_p = __value(p)
    cache.partials_p = ForwardDiff.partials(p)
    return cache
end

for algType in (
    Nothing, AbstractNonlinearSolveAlgorithm, GeneralizedDFSane,
    SimpleNonlinearSolve.AbstractSimpleNonlinearSolveAlgorithm,
    GeneralizedFirstOrderAlgorithm, ApproximateJacobianSolveAlgorithm,
    LeastSquaresOptimJL, FastLevenbergMarquardtJL, CMINPACK, NLsolveJL, NLSolversJL,
    SpeedMappingJL, FixedPointAccelerationJL, SIAMFANLEquationsJL,
    NonlinearSolvePolyAlgorithm{:NLLS, <:Any}, NonlinearSolvePolyAlgorithm{:NLS, <:Any}
)
    @eval function SciMLBase.__init(
            prob::DualNonlinearProblem, alg::$(algType), args...; kwargs...)
        p = __value(prob.p)
        newprob = NonlinearProblem(prob.f, __value(prob.u0), p; prob.kwargs...)
        cache = init(newprob, alg, args...; kwargs...)
        return NonlinearSolveForwardDiffCache(
            cache, newprob, alg, prob.p, p, ForwardDiff.partials(prob.p))
    end
end

function SciMLBase.solve!(cache::NonlinearSolveForwardDiffCache)
    sol = solve!(cache.cache)
    prob = cache.prob

    uu = sol.u
    Jₚ = nonlinearsolve_∂f_∂p(prob, prob.f, uu, cache.values_p)
    Jᵤ = nonlinearsolve_∂f_∂u(prob, prob.f, uu, cache.values_p)

    z_arr = -Jᵤ \ Jₚ

    sumfun = ((z, p),) -> map(zᵢ -> zᵢ * ForwardDiff.partials(p), z)
    if cache.p isa Number
        partials = sumfun((z_arr, cache.p))
    else
        partials = sum(sumfun, zip(eachcol(z_arr), cache.p))
    end

    dual_soln = nonlinearsolve_dual_solution(sol.u, partials, cache.p)
    return SciMLBase.build_solution(
        prob, cache.alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original)
end

@inline __value(x) = x
@inline __value(x::Dual) = ForwardDiff.value(x)
@inline __value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)
