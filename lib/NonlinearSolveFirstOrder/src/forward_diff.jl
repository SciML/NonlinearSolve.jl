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

function SciMLBase.__init(
        prob::DualAbstractNonlinearProblem, alg::GeneralizedFirstOrderAlgorithm, args...; kwargs...
)
    p = NonlinearSolveBase.nodual_value(prob.p)
    newprob = SciMLBase.remake(prob; u0 = NonlinearSolveBase.nodual_value(prob.u0), p)
    cache = init(newprob, alg, args...; kwargs...)
    return NonlinearSolveForwardDiffCache(
        cache, newprob, alg, prob.p, p, ForwardDiff.partials(prob.p)
    )
end

function SciMLBase.__solve(
        prob::DualAbstractNonlinearProblem, alg::GeneralizedFirstOrderAlgorithm, args...; kwargs...
)
    sol,
    partials = NonlinearSolveBase.nonlinearsolve_forwarddiff_solve(
        prob, alg, args...; kwargs...
    )
    dual_soln = NonlinearSolveBase.nonlinearsolve_dual_solution(sol.u, partials, prob.p)
    return SciMLBase.build_solution(
        prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original
    )
end
