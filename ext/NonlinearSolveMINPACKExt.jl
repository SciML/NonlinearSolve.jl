module NonlinearSolveMINPACKExt

using FastClosures: @closure
using MINPACK: MINPACK

using NonlinearSolveBase: NonlinearSolveBase
using NonlinearSolve: NonlinearSolve, CMINPACK
using SciMLBase: SciMLBase, NonlinearLeastSquaresProblem, NonlinearProblem, ReturnCode

function SciMLBase.__solve(
        prob::Union{NonlinearLeastSquaresProblem, NonlinearProblem}, alg::CMINPACK, args...;
        abstol = nothing, maxiters = 1000, alias_u0::Bool = false,
        show_trace::Val = Val(false), store_trace::Val = Val(false),
        termination_condition = nothing, kwargs...
    )
    NonlinearSolveBase.assert_extension_supported_termination_condition(
        termination_condition, alg
    )

    f_wrapped!, u0,
        resid = NonlinearSolveBase.construct_extension_function_wrapper(
        prob; alias_u0
    )
    resid_size = size(resid)
    f! = @closure (du, u) -> (f_wrapped!(du, u); Cint(0))
    m = length(resid)

    method = ifelse(
        alg.method === :auto,
        ifelse(prob isa NonlinearLeastSquaresProblem, :lm, :hybr), alg.method
    )

    show_trace = show_trace isa Val{true}
    tracing = store_trace isa Val{true}
    tol = NonlinearSolveBase.get_tolerance(abstol, eltype(u0))

    if alg.autodiff === missing && prob.f.jac === nothing
        original = MINPACK.fsolve(
            f!, u0, m; tol, show_trace, tracing, method, iterations = maxiters
        )
    else
        autodiff = alg.autodiff === missing ? nothing : alg.autodiff
        jac_wrapped! = NonlinearSolveBase.construct_extension_jac(
            prob, alg, u0, resid; autodiff
        )
        jac! = @closure (J, u) -> (jac_wrapped!(J, u); Cint(0))
        original = MINPACK.fsolve(
            f!, jac!, u0, m; tol, show_trace, tracing, method, iterations = maxiters
        )
    end

    u = original.x
    resid = original.f
    objective = maximum(abs, resid)
    retcode = ifelse(objective ≤ tol, ReturnCode.Success, ReturnCode.Failure)

    # These are only meaningful if `store_trace = Val(true)`
    stats = SciMLBase.NLStats(
        original.trace.f_calls, original.trace.g_calls,
        original.trace.g_calls, original.trace.g_calls, -1
    )

    u = prob.u0 isa Number ? original.x[1] : reshape(original.x, size(prob.u0))
    resid = prob.u0 isa Number ? resid[1] : reshape(resid, resid_size)
    return SciMLBase.build_solution(prob, alg, u, resid; retcode, original, stats)
end

end
