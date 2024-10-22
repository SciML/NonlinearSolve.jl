module NonlinearSolveNLsolveExt

using LineSearches: Static
using NonlinearSolveBase: NonlinearSolveBase, get_tolerance
using NonlinearSolve: NonlinearSolve, NLsolveJL, TraceMinimal
using NLsolve: NLsolve, OnceDifferentiable, nlsolve
using SciMLBase: SciMLBase, NonlinearProblem, ReturnCode

function SciMLBase.__solve(
        prob::NonlinearProblem, alg::NLsolveJL, args...; abstol = nothing,
        maxiters = 1000, alias_u0::Bool = false, termination_condition = nothing,
        store_trace::Val{StT} = Val(false), show_trace::Val{ShT} = Val(false),
        trace_level = TraceMinimal(), kwargs...) where {StT, ShT}
    NonlinearSolve.__test_termination_condition(termination_condition, :NLsolveJL)

    f!, u0, resid = NonlinearSolve.__construct_extension_f(prob; alias_u0)

    if prob.f.jac === nothing && alg.autodiff isa Symbol
        df = OnceDifferentiable(f!, u0, resid; alg.autodiff)
    else
        jac! = NonlinearSolve.__construct_extension_jac(prob, alg, u0, resid; alg.autodiff)
        if prob.f.jac_prototype === nothing
            J = similar(
                u0, promote_type(eltype(u0), eltype(resid)), length(u0), length(resid))
        else
            J = zero(prob.f.jac_prototype)
        end
        df = OnceDifferentiable(f!, jac!, vec(u0), vec(resid), J)
    end

    abstol = get_tolerance(abstol, eltype(u0))
    show_trace = ShT
    store_trace = StT
    extended_trace = !(trace_level isa TraceMinimal)

    linesearch = alg.linesearch === missing ? Static() : alg.linesearch

    original = nlsolve(df, vec(u0); ftol = abstol, iterations = maxiters, alg.method,
        store_trace, extended_trace, linesearch, alg.linsolve, alg.factor,
        alg.autoscale, alg.m, alg.beta, show_trace)

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
