module NonlinearSolveLeastSquaresOptimExt

using LeastSquaresOptim: LeastSquaresOptim

using NonlinearSolveBase: NonlinearSolveBase, TraceMinimal
using NonlinearSolve: NonlinearSolve, LeastSquaresOptimJL
using SciMLBase: SciMLBase, AbstractNonlinearProblem, ReturnCode

const LSO = LeastSquaresOptim

function SciMLBase.__solve(
        prob::AbstractNonlinearProblem, alg::LeastSquaresOptimJL, args...;
        alias_u0 = false, abstol = nothing, reltol = nothing, maxiters = 1000,
        trace_level = TraceMinimal(), termination_condition = nothing,
        show_trace::Val = Val(false), store_trace::Val = Val(false), kwargs...
)
    NonlinearSolveBase.assert_extension_supported_termination_condition(
        termination_condition, alg
    )

    f!, u, resid = NonlinearSolveBase.construct_extension_function_wrapper(prob; alias_u0)
    abstol = NonlinearSolveBase.get_tolerance(abstol, eltype(u))
    reltol = NonlinearSolveBase.get_tolerance(reltol, eltype(u))

    if prob.f.jac === nothing && alg.autodiff isa Symbol
        lsoprob = LSO.LeastSquaresProblem(;
            x = u, f!, y = resid, alg.autodiff, J = prob.f.jac_prototype,
            output_length = length(resid)
        )
    else
        g! = NonlinearSolveBase.construct_extension_jac(prob, alg, u, resid; alg.autodiff)
        lsoprob = LSO.LeastSquaresProblem(;
            x = u, f!, y = resid, g!, J = prob.f.jac_prototype,
            output_length = length(resid)
        )
    end

    linsolve = alg.ls === :qr ? LSO.QR() :
               (alg.ls === :cholesky ? LSO.Cholesky() :
                (alg.ls === :lsmr ? LSO.LSMR() : nothing))

    lso_solver = if alg.alg === :lm
        LSO.LevenbergMarquardt(linsolve)
    elseif alg.alg === :dogleg
        LSO.Dogleg(linsolve)
    else
        throw(ArgumentError("Unknown LeastSquaresOptim Algorithm: $(Meta.quot(alg.alg))"))
    end

    allocated_prob = LSO.LeastSquaresProblemAllocated(lsoprob, lso_solver(alg))
    res = LSO.optimize!(
        allocated_prob;
        x_tol = reltol, f_tol = abstol, g_tol = abstol, iterations = maxiters,
        show_trace = show_trace isa Val{true}, store_trace = store_trace isa Val{true},
        show_every = trace_level.print_frequency
    )

    retcode = res.x_converged || res.f_converged || res.g_converged ? ReturnCode.Success :
              (res.iterations ≥ maxiters ? ReturnCode.MaxIters :
               ReturnCode.ConvergenceFailure)
    stats = SciMLBase.NLStats(res.f_calls, res.g_calls, -1, -1, res.iterations)

    f!(resid, res.minimizer)

    return SciMLBase.build_solution(
        prob, alg, res.minimizer, resid; retcode, original = res, stats
    )
end

end
