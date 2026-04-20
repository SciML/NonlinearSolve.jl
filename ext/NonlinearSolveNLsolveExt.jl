module NonlinearSolveNLsolveExt

using LineSearches: Static
using NLsolve: NLsolve, NonDifferentiable, OnceDifferentiable, nlsolve

using NonlinearSolveBase: NonlinearSolveBase, Utils, TraceMinimal, is_fw_wrapped, get_raw_f
using NonlinearSolve: NonlinearSolve, NLsolveJL
using SciMLBase: SciMLBase, NonlinearProblem, ReturnCode
using Setfield: @set

function SciMLBase.__solve(
        prob::NonlinearProblem, alg::NLsolveJL, args...;
        abstol = nothing, maxiters = 1000, alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = false),
        termination_condition = nothing, trace_level = TraceMinimal(),
        store_trace::Val = Val(false), show_trace::Val = Val(false), kwargs...
    )
    # Unwrap AutoSpecialize — external packages do their own AD
    if is_fw_wrapped(prob.f.f)
        prob = @set prob.f.f = get_raw_f(prob.f.f)
    end

    if haskey(kwargs, :alias_u0)
        alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = kwargs[:alias_u0])
    end
    alias_u0 = alias.alias_u0
    NonlinearSolveBase.assert_extension_supported_termination_condition(
        termination_condition, alg
    )

    f!, u0, resid = NonlinearSolveBase.construct_extension_function_wrapper(prob; alias_u0)

    # Anderson and Broyden do not use a Jacobian — use NonDifferentiable to avoid
    # allocating a dense N×N Jacobian (matches NLsolve.nlsolve's own behavior).
    if alg.method in (:anderson, :broyden)
        df = NonDifferentiable(f!, Utils.safe_vec(u0), Utils.safe_vec(resid); inplace = true)
    elseif prob.f.jac === nothing && alg.autodiff isa Symbol
        df = OnceDifferentiable(f!, u0, resid; alg.autodiff)
    else
        autodiff = alg.autodiff isa Symbol ? nothing : alg.autodiff
        jac! = NonlinearSolveBase.construct_extension_jac(prob, alg, u0, resid; autodiff)
        if prob.f.jac_prototype === nothing
            J = similar(
                u0, promote_type(eltype(u0), eltype(resid)), length(u0), length(resid)
            )
        else
            J = zero(prob.f.jac_prototype)
        end
        df = OnceDifferentiable(f!, jac!, Utils.safe_vec(u0), Utils.safe_vec(resid), J)
    end

    abstol = NonlinearSolveBase.get_tolerance(abstol, eltype(u0))
    show_trace = show_trace isa Val{true}
    store_trace = store_trace isa Val{true}
    extended_trace = !(trace_level.trace_mode isa Val{:minimal})

    linesearch = alg.linesearch === missing ? Static() : alg.linesearch

    original = nlsolve(
        df, vec(u0);
        ftol = abstol, iterations = maxiters, alg.method, store_trace, extended_trace,
        linesearch, alg.linsolve, alg.factor, alg.autoscale, alg.m, alg.beta, show_trace
    )

    f!(vec(resid), original.zero)
    u = prob.u0 isa Number ? original.zero[1] : reshape(original.zero, size(prob.u0))
    resid = prob.u0 isa Number ? resid[1] : resid

    retcode = original.x_converged || original.f_converged ? ReturnCode.Success :
        ReturnCode.Failure
    stats = SciMLBase.NLStats(
        original.f_calls, original.g_calls, original.g_calls,
        original.g_calls, original.iterations
    )

    return SciMLBase.build_solution(prob, alg, u, resid; retcode, original, stats)
end

end
