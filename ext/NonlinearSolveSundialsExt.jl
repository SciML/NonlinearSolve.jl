module NonlinearSolveSundialsExt

using NonlinearSolveBase: NonlinearSolveBase, nonlinearsolve_forwarddiff_solve,
                          nonlinearsolve_dual_solution
using NonlinearSolve: DualNonlinearProblem
using SciMLBase: SciMLBase
using Sundials: KINSOL

function SciMLBase.__solve(prob::DualNonlinearProblem, alg::KINSOL, args...; kwargs...)
    sol, partials = nonlinearsolve_forwarddiff_solve(prob, alg, args...; kwargs...)
    dual_soln = nonlinearsolve_dual_solution(sol.u, partials, prob.p)
    return SciMLBase.build_solution(
        prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original)
end

end
