module NonlinearSolveMINPACKExt

using NonlinearSolve, DiffEqBase, SciMLBase
using MINPACK
import FastClosures: @closure

function SciMLBase.__solve(prob::Union{NonlinearProblem{uType, iip},
            NonlinearLeastSquaresProblem{uType, iip}}, alg::CMINPACK, args...;
        abstol = nothing, maxiters = 1000, alias_u0::Bool = false,
        show_trace::Val{ShT} = Val(false), store_trace::Val{StT} = Val(false),
        termination_condition = nothing, kwargs...) where {uType, iip, ShT, StT}
    @assert (termination_condition ===
             nothing)||(termination_condition isa AbsNormTerminationMode) "CMINPACK does not support termination conditions!"

    f!_, u0 = NonlinearSolve.__construct_f(prob; alias_u0)
    f! = @closure (du, u) -> (f!_(du, u); Cint(0))

    resid = NonlinearSolve.evaluate_f(prob, prob.u0)
    m = length(resid)

    method = ifelse(alg.method === :auto,
        ifelse(prob isa NonlinearLeastSquaresProblem, :lm, :hybr), alg.method)

    show_trace = alg.show_trace || ShT
    tracing = alg.tracing || StT
    tol = NonlinearSolve.DEFAULT_TOLERANCE(abstol, eltype(u0))

    jac!_ = NonlinearSolve.__construct_jac(prob, alg, u0)

    if jac!_ === nothing
        original = MINPACK.fsolve(f!, u0, m; tol, show_trace, tracing, method,
            iterations = maxiters)
    else
        jac! = @closure((J, u)->(jac!_(J, u); Cint(0)))
        original = MINPACK.fsolve(f!, jac!, u0, m; tol, show_trace, tracing, method,
            iterations = maxiters)
    end

    u = original.x
    resid_ = original.f
    objective = maximum(abs, resid_)
    retcode = ifelse(objective ≤ tol, ReturnCode.Success, ReturnCode.Failure)

    # These are only meaningful if `store_trace = Val(true)`
    stats = SciMLBase.NLStats(original.trace.f_calls, original.trace.g_calls,
        original.trace.g_calls, original.trace.g_calls, -1)

    u_ = prob.u0 isa Number ? original.x[1] : reshape(original.x, size(prob.u0))
    resid_ = prob.u0 isa Number ? resid_[1] : reshape(resid_, size(resid))
    return SciMLBase.build_solution(prob, alg, u_, resid_; retcode, original, stats)
end

end
