module NonlinearSolveFixedPointAccelerationExt

using FixedPointAcceleration: FixedPointAcceleration, fixed_point

using NonlinearSolveBase: NonlinearSolveBase
using NonlinearSolve: NonlinearSolve, FixedPointAccelerationJL
using SciMLBase: SciMLBase, NonlinearProblem, ReturnCode

function SciMLBase.__solve(
        prob::NonlinearProblem, alg::FixedPointAccelerationJL, args...;
        abstol = nothing, maxiters = 1000, alias_u0::Bool = false,
        show_trace::Val = Val(false), termination_condition = nothing, kwargs...
)
    NonlinearSolveBase.assert_extension_supported_termination_condition(
        termination_condition, alg
    )

    f, u0, resid = NonlinearSolveBase.construct_extension_function_wrapper(
        prob; alias_u0, make_fixed_point = Val(true), force_oop = Val(true)
    )

    tol = NonlinearSolveBase.get_tolerance(abstol, eltype(u0))

    sol = fixed_point(
        f, u0; Algorithm = alg.algorithm, MaxIter = maxiters, MaxM = alg.m,
        ConvergenceMetricThreshold = tol, ExtrapolationPeriod = alg.extrapolation_period,
        Dampening = alg.dampening, PrintReports = show_trace isa Val{true},
        ReplaceInvalids = alg.replace_invalids,
        ConditionNumberThreshold = alg.condition_number_threshold, quiet_errors = true
    )

    if sol.FixedPoint_ === missing
        u0 = prob.u0 isa Number ? u0[1] : u0
        resid = NonlinearSolveBase.Utils.evaluate_f(prob, u0)
        res = u0
        converged = false
    else
        res = prob.u0 isa Number ? first(sol.FixedPoint_) :
              reshape(sol.FixedPoint_, size(prob.u0))
        resid = NonlinearSolveBase.Utils.evaluate_f(prob, res)
        converged = maximum(abs, resid) ≤ tol
    end

    return SciMLBase.build_solution(
        prob, alg, res, resid; original = sol,
        retcode = converged ? ReturnCode.Success : ReturnCode.Failure,
        stats = SciMLBase.NLStats(sol.Iterations_, 0, 0, 0, sol.Iterations_)
    )
end

end
