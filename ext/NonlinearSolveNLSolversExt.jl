module NonlinearSolveNLSolversExt

using ADTypes, FastClosures, NonlinearSolve, NLSolvers, SciMLBase, LinearAlgebra
using FiniteDiff, ForwardDiff

function SciMLBase.__solve(prob::NonlinearProblem, alg::NLSolversJL, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000,
        alias_u0::Bool = false, termination_condition = nothing, kwargs...)
    NonlinearSolve.__test_termination_condition(termination_condition, :NLSolversJL)

    abstol = NonlinearSolve.DEFAULT_TOLERANCE(abstol, eltype(prob.u0))
    reltol = NonlinearSolve.DEFAULT_TOLERANCE(reltol, eltype(prob.u0))

    options = NEqOptions(; maxiter = maxiters, f_abstol = abstol, f_reltol = reltol)

    if prob.u0 isa Number
        f_scalar = @closure x -> prob.f(x, prob.p)

        if alg.autodiff === nothing
            if ForwardDiff.can_dual(typeof(prob.u0))
                autodiff_concrete = :forwarddiff
            else
                autodiff_concrete = :finitediff
            end
        else
            if alg.autodiff isa AutoForwardDiff || alg.autodiff isa AutoPolyesterForwardDiff
                autodiff_concrete = :forwarddiff
            elseif alg.autodiff isa AutoFiniteDiff
                autodiff_concrete = :finitediff
            else
                error("Only ForwardDiff or FiniteDiff autodiff is supported.")
            end
        end

        if autodiff_concrete === :forwarddiff
            fj_scalar = @closure (Jx, x) -> begin
                T = typeof(ForwardDiff.Tag(prob.f, eltype(x)))
                x_dual = ForwardDiff.Dual{T}(x, one(x))
                y = prob.f(x_dual, prob.p)
                return ForwardDiff.value(y), ForwardDiff.extract_derivative(T, y)
            end
        else
            fj_scalar = @closure (Jx, x) -> begin
                _f = Base.Fix2(prob.f, prob.p)
                return _f(x), FiniteDiff.finite_difference_derivative(_f, x)
            end
        end

        prob_obj = NLSolvers.ScalarObjective(; f = f_scalar, fg = fj_scalar)
        prob_nlsolver = NEqProblem(prob_obj; inplace = false)
        res = NLSolvers.solve(prob_nlsolver, prob.u0, alg.method, options)

        retcode = ifelse(norm(res.info.best_residual, Inf) ≤ abstol,
            ReturnCode.Success, ReturnCode.MaxIters)
        stats = SciMLBase.NLStats(-1, -1, -1, -1, res.info.iter)

        return SciMLBase.build_solution(
            prob, alg, res.info.solution, res.info.best_residual;
            retcode, original = res, stats)
    end

    f!, u0, resid = NonlinearSolve.__construct_extension_f(prob; alias_u0)

    jac! = NonlinearSolve.__construct_extension_jac(prob, alg, u0, resid; alg.autodiff)

    FJ_vector! = @closure (Fx, Jx, x) -> begin
        f!(Fx, x)
        jac!(Jx, x)
        return Fx, Jx
    end

    prob_obj = NLSolvers.VectorObjective(; F = f!, FJ = FJ_vector!)
    prob_nlsolver = NEqProblem(prob_obj)

    res = NLSolvers.solve(prob_nlsolver, u0, alg.method, options)

    retcode = ifelse(
        norm(res.info.best_residual, Inf) ≤ abstol, ReturnCode.Success, ReturnCode.MaxIters)
    stats = SciMLBase.NLStats(-1, -1, -1, -1, res.info.iter)

    return SciMLBase.build_solution(prob, alg, res.info.solution, res.info.best_residual;
        retcode, original = res, stats)
end

end
