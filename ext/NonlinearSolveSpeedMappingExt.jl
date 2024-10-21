module NonlinearSolveSpeedMappingExt

using NonlinearSolve: NonlinearSolve, SpeedMappingJL
using SciMLBase: SciMLBase, NonlinearProblem, ReturnCode
using SpeedMapping: speedmapping

function SciMLBase.__solve(prob::NonlinearProblem, alg::SpeedMappingJL, args...;
        abstol = nothing, maxiters = 1000, alias_u0::Bool = false,
        maxtime = nothing, store_trace::Val{store_info} = Val(false),
        termination_condition = nothing, kwargs...) where {store_info}
    NonlinearSolve.__test_termination_condition(termination_condition, :SpeedMappingJL)

    m!, u, resid = NonlinearSolve.__construct_extension_f(
        prob; alias_u0, make_fixed_point = Val(true))
    tol = NonlinearSolve.DEFAULT_TOLERANCE(abstol, eltype(u))

    time_limit = ifelse(maxtime === nothing, 1000, maxtime)

    sol = speedmapping(u; m!, tol, Lp = Inf, maps_limit = maxiters, alg.orders,
        alg.check_obj, store_info, alg.σ_min, alg.stabilize, time_limit)
    res = prob.u0 isa Number ? first(sol.minimizer) : sol.minimizer
    resid = NonlinearSolve.evaluate_f(prob, res)

    return SciMLBase.build_solution(prob, alg, res, resid; original = sol,
        retcode = sol.converged ? ReturnCode.Success : ReturnCode.Failure,
        stats = SciMLBase.NLStats(sol.maps, 0, 0, 0, sol.maps))
end

end
