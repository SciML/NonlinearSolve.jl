module NonlinearSolveNLSolversExt

using DifferentiationInterface: DifferentiationInterface, Constant
using FastClosures: @closure

using NLSolvers: NLSolvers, NEqOptions, NEqProblem

using NonlinearSolveBase: NonlinearSolveBase
using NonlinearSolve: NonlinearSolve, NLSolversJL
using SciMLBase: SciMLBase, NonlinearProblem, ReturnCode

const DI = DifferentiationInterface

function SciMLBase.__solve(
        prob::NonlinearProblem, alg::NLSolversJL, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000, alias_u0::Bool = false,
        termination_condition = nothing, kwargs...
)
    NonlinearSolveBase.assert_extension_supported_termination_condition(
        termination_condition, alg
    )

    abstol = NonlinearSolveBase.get_tolerance(abstol, eltype(prob.u0))
    reltol = NonlinearSolveBase.get_tolerance(reltol, eltype(prob.u0))

    options = NEqOptions(; maxiter = maxiters, f_abstol = abstol, f_reltol = reltol)

    if prob.u0 isa Number
        f_scalar = Base.Fix2(prob.f, prob.p)
        autodiff = NonlinearSolveBase.select_jacobian_autodiff(prob, alg.autodiff)

        prep = DifferentiationInterface.prepare_derivative(
            prob.f, autodiff, prob.u0, Constant(prob.p)
        )

        fj_scalar = @closure (Jx, x) -> begin
            return DifferentiationInterface.value_and_derivative(
                prob.f, prep, autodiff, x, Constant(prob.p)
            )
        end

        prob_obj = NLSolvers.ScalarObjective(; f = f_scalar, fg = fj_scalar)
        prob_nlsolver = NEqProblem(prob_obj; inplace = false)
        res = NLSolvers.solve(prob_nlsolver, prob.u0, alg.method, options)

        retcode = ifelse(
            maximum(abs, res.info.best_residual) ≤ abstol,
            ReturnCode.Success, ReturnCode.MaxIters
        )
        stats = SciMLBase.NLStats(-1, -1, -1, -1, res.info.iter)

        return SciMLBase.build_solution(
            prob, alg, res.info.solution, res.info.best_residual;
            retcode, original = res, stats
        )
    end

    f!, u0, resid = NonlinearSolveBase.construct_extension_function_wrapper(prob; alias_u0)
    jac! = NonlinearSolveBase.construct_extension_jac(prob, alg, u0, resid; alg.autodiff)

    FJ_vector! = @closure (Fx, Jx, x) -> begin
        f!(Fx, x)
        jac!(Jx, x)
        return Fx, Jx
    end

    prob_obj = NLSolvers.VectorObjective(; F = f!, FJ = FJ_vector!)
    prob_nlsolver = NEqProblem(prob_obj)

    res = NLSolvers.solve(prob_nlsolver, u0, alg.method, options)

    retcode = ifelse(
        maximum(abs, res.info.best_residual) ≤ abstol,
        ReturnCode.Success, ReturnCode.MaxIters
    )
    stats = SciMLBase.NLStats(-1, -1, -1, -1, res.info.iter)

    return SciMLBase.build_solution(
        prob, alg, res.info.solution, res.info.best_residual;
        retcode, original = res, stats
    )
end

end
