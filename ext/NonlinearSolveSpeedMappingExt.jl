module NonlinearSolveSpeedMappingExt

using SpeedMapping: speedmapping

using NonlinearSolveBase: NonlinearSolveBase
using NonlinearSolve: NonlinearSolve, SpeedMappingJL
using SciMLBase: SciMLBase, NonlinearProblem, ReturnCode

function SciMLBase.__solve(
        prob::NonlinearProblem, alg::SpeedMappingJL, args...;
        abstol = nothing, maxiters = 1000, alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = false),
        maxtime = nothing, store_trace::Val = Val(false),
        termination_condition = nothing, kwargs...
)
    alias_u0 = alias.alias_u0
    NonlinearSolveBase.assert_extension_supported_termination_condition(
        termination_condition, alg
    )

    m!, u,
    resid = NonlinearSolveBase.construct_extension_function_wrapper(
        prob; alias_u0, make_fixed_point = Val(true)
    )
    tol = NonlinearSolveBase.get_tolerance(abstol, eltype(u))

    time_limit = ifelse(maxtime === nothing, 1000, maxtime)

    sol = speedmapping(
        u; m!, tol, Lp = Inf, maps_limit = maxiters, alg.orders,
        alg.check_obj, store_info = store_trace isa Val{true}, alg.σ_min, alg.stabilize,
        time_limit
    )
    res = prob.u0 isa Number ? first(sol.minimizer) : sol.minimizer
    resid = NonlinearSolveBase.Utils.evaluate_f(prob, res)

    return SciMLBase.build_solution(
        prob, alg, res, resid;
        original = sol, stats = SciMLBase.NLStats(sol.maps, 0, 0, 0, sol.maps),
        retcode = ifelse(sol.converged, ReturnCode.Success, ReturnCode.Failure)
    )
end

end
