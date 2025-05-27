module BracketingNonlinearSolveForwardDiffExt

using CommonSolve: CommonSolve
using ForwardDiff: Dual
using NonlinearSolveBase: nonlinearsolve_forwarddiff_solve, nonlinearsolve_dual_solution
using SciMLBase: SciMLBase, IntervalNonlinearProblem

using BracketingNonlinearSolve: Bisection, Brent, Alefeld, Falsi, ITP, Ridder

const DualIntervalNonlinearProblem{T, V, P} = IntervalNonlinearProblem{
    uType, iip, <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}}
} where {uType, iip}

for algT in (Bisection, Brent, Alefeld, Falsi, ITP, Ridder)
    @eval function CommonSolve.solve(
            prob::DualIntervalNonlinearProblem{T, V, P}, alg::$(algT), args...;
            kwargs...
    ) where {T, V, P}
        sol, partials = nonlinearsolve_forwarddiff_solve(prob, alg, args...; kwargs...)
        dual_soln = nonlinearsolve_dual_solution(sol.u, partials, prob.p)
        return SciMLBase.build_solution(
            prob, alg, dual_soln, sol.resid;
            sol.retcode, sol.stats, sol.original,
            left = Dual{T, V, P}(sol.left, partials),
            right = Dual{T, V, P}(sol.right, partials)
        )
    end
end

end
