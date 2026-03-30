module NonlinearSolveSundialsExt

using Sundials: KINSOL

using CommonSolve: CommonSolve
using NonlinearSolveBase: NonlinearSolveBase, nonlinearsolve_forwarddiff_solve,
    nonlinearsolve_dual_solution, is_fw_wrapped, get_raw_f
using NonlinearSolve: NonlinearSolve, DualNonlinearProblem
using SciMLBase: SciMLBase, remake
using Setfield: @set

function SciMLBase.__solve(prob::DualNonlinearProblem, alg::KINSOL, args...; kwargs...)
    # Unwrap AutoSpecialize — external packages do their own AD
    if is_fw_wrapped(prob.f.f)
        prob = @set prob.f.f = get_raw_f(prob.f.f)
    end

    sol, partials = nonlinearsolve_forwarddiff_solve(prob, alg, args...; kwargs...)
    dual_soln = nonlinearsolve_dual_solution(sol.u, partials, prob.p)
    return SciMLBase.build_solution(
        prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original
    )
end

function SciMLBase.__init(prob::DualNonlinearProblem, alg::KINSOL, args...; kwargs...)
    p = NonlinearSolveBase.nodual_value(prob.p)
    newprob = SciMLBase.remake(prob; u0 = NonlinearSolveBase.nodual_value(prob.u0), p)
    cache = CommonSolve.init(newprob, alg, args...; kwargs...)
    return NonlinearSolveForwardDiffCache(
        cache, newprob, alg, prob.p, p, ForwardDiff.partials(prob.p)
    )
end

end
