module NonlinearSolveSpeedMappingExt

using NonlinearSolve, SpeedMapping, DiffEqBase, SciMLBase

function SciMLBase.__solve(prob::NonlinearProblem, alg::SpeedMappingJL, args...;
        abstol = nothing, maxiters = 1000, alias_u0::Bool = false,
        store_trace::Val{store_info} = Val(false), termination_condition = nothing,
        kwargs...) where {store_info}
    @assert (termination_condition ===
             nothing)||(termination_condition isa AbsNormTerminationMode) "SpeedMappingJL does not support termination conditions!"

    m!, u0 = NonlinearSolve.__construct_f(prob; alias_u0, make_fixed_point = Val(true),
        can_handle_arbitrary_dims = Val(true))

    tol = NonlinearSolve.DEFAULT_TOLERANCE(abstol, eltype(u0))

    sol = speedmapping(u0; m!, tol, Lp = Inf, maps_limit = maxiters, alg.orders,
        alg.check_obj, store_info, alg.σ_min, alg.stabilize)
    res = prob.u0 isa Number ? first(sol.minimizer) : sol.minimizer
    resid = NonlinearSolve.evaluate_f(prob, res)

    return SciMLBase.build_solution(prob, alg, res, resid;
        retcode = sol.converged ? ReturnCode.Success : ReturnCode.Failure,
        stats = SciMLBase.NLStats(sol.maps, 0, 0, 0, sol.maps), original = sol)
end

end
