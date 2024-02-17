module NonlinearSolveNLSolversExt

using NonlinearSolve, NLSolversJL, SciMLBase

function SciMLBase.__solve(prob::NonlinearProblem, alg::NLSolversJL, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000, alias_u0::Bool = false,
        termination_condition = nothing, kwargs...) where {StT, ShT}
    NonlinearSolve.__test_termination_condition(termination_condition, :NLSolversJL)

    if prob.u0 isa Number
        error("Scalar Inputs for NLsolversJL is not yet handled.")
    end

    f!, u0, resid = NonlinearSolve.__construct_extension_f(prob; alias_u0)

#     if prob.f.jac === nothing && alg.autodiff isa Symbol
#         df = OnceDifferentiable(f!, u0, resid; alg.autodiff)
#     else
#         jac! = NonlinearSolve.__construct_extension_jac(prob, alg, u0, resid; alg.autodiff)
#         if prob.f.jac_prototype === nothing
#             J = similar(u0, promote_type(eltype(u0), eltype(resid)), length(u0),
#                 length(resid))
#         else
#             J = zero(prob.f.jac_prototype)
#         end
#         df = OnceDifferentiable(f!, jac!, vec(u0), vec(resid), J)
#     end

#     abstol = NonlinearSolve.DEFAULT_TOLERANCE(abstol, eltype(u0))
#     show_trace = ShT || alg.show_trace
#     store_trace = StT || alg.store_trace
#     extended_trace = !(trace_level isa TraceMinimal) || alg.extended_trace

#     original = nlsolve(df, vec(u0); ftol = abstol, iterations = maxiters, alg.method,
#         store_trace, extended_trace, alg.linesearch, alg.linsolve, alg.factor,
#         alg.autoscale, alg.m, alg.beta, show_trace)

#     f!(vec(resid), original.zero)
#     u = prob.u0 isa Number ? original.zero[1] : reshape(original.zero, size(prob.u0))
#     resid = prob.u0 isa Number ? resid[1] : resid

#     retcode = original.x_converged || original.f_converged ? ReturnCode.Success :
#               ReturnCode.Failure
#     stats = SciMLBase.NLStats(original.f_calls, original.g_calls, original.g_calls,
#         original.g_calls, original.iterations)

#     return SciMLBase.build_solution(prob, alg, u, resid; retcode, original, stats)
end

end
