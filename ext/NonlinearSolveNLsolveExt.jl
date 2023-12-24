module NonlinearSolveNLsolveExt

using NonlinearSolve, NLsolve, DiffEqBase, SciMLBase

function SciMLBase.__solve(prob::NonlinearProblem, alg::NLsolveJL, args...;
        abstol = nothing, maxiters = 1000, alias_u0::Bool = false,
        termination_condition = nothing, kwargs...)
    @assert (termination_condition ===
             nothing)||(termination_condition isa AbsNormTerminationMode) "NLsolveJL does not support termination conditions!"

    f!, u0 = NonlinearSolve.__construct_f(prob; alias_u0)

    # unwrapping alg params
    (; method, autodiff, store_trace, extended_trace, linesearch, linsolve, factor,
    autoscale, m, beta, show_trace) = alg

    if prob.u0 isa Number
        resid = [NonlinearSolve.evaluate_f(prob, first(u0))]
    else
        resid = NonlinearSolve.evaluate_f(prob, u0)
    end

    jac! = NonlinearSolve.__construct_jac(prob, alg, u0)

    if jac! === nothing
        df = OnceDifferentiable(f!, vec(u0), vec(resid); autodiff)
    else
        if prob.f.jac_prototype !== nothing
            J = zero(prob.f.jac_prototype)
            df = OnceDifferentiable(f!, jac!, vec(u0), vec(resid), J)
        else
            df = OnceDifferentiable(f!, jac!, vec(u0), vec(resid))
        end
    end

    abstol = NonlinearSolve.DEFAULT_TOLERANCE(abstol, eltype(u0))

    original = nlsolve(df, vec(u0); ftol = abstol, iterations = maxiters, method,
        store_trace, extended_trace, linesearch, linsolve, factor, autoscale, m, beta,
        show_trace)

    f!(vec(resid), original.zero)
    u = prob.u0 isa Number ? original.zero[1] : reshape(original.zero, size(prob.u0))
    resid = prob.u0 isa Number ? resid[1] : resid

    retcode = original.x_converged || original.f_converged ? ReturnCode.Success :
              ReturnCode.Failure
    stats = SciMLBase.NLStats(original.f_calls, original.g_calls, original.g_calls,
        original.g_calls, original.iterations)

    return SciMLBase.build_solution(prob, alg, u, resid; retcode, original, stats)
end

end
