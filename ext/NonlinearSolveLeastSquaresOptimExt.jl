module NonlinearSolveLeastSquaresOptimExt

using NonlinearSolve, SciMLBase
import ConcreteStructs: @concrete
import LeastSquaresOptim as LSO

@inline function _lso_solver(::LeastSquaresOptimJL{alg, ls}) where {alg, ls}
    linsolve = ls === :qr ? LSO.QR() :
               (ls === :cholesky ? LSO.Cholesky() : (ls === :lsmr ? LSO.LSMR() : nothing))
    if alg === :lm
        return LSO.LevenbergMarquardt(linsolve)
    elseif alg === :dogleg
        return LSO.Dogleg(linsolve)
    else
        throw(ArgumentError("Unknown LeastSquaresOptim Algorithm: $alg"))
    end
end

@concrete struct LeastSquaresOptimJLCache
    prob
    alg
    allocated_prob
    kwargs
end

function Base.show(io::IO, cache::LeastSquaresOptimJLCache)
    print(io, "LeastSquaresOptimJLCache()")
end

function SciMLBase.reinit!(cache::LeastSquaresOptimJLCache, args...; kwargs...)
    error("Reinitialization not supported for LeastSquaresOptimJL.")
end

function SciMLBase.__init(prob::Union{NonlinearLeastSquaresProblem, NonlinearProblem},
        alg::LeastSquaresOptimJL, args...; alias_u0 = false, abstol = nothing,
        show_trace::Val{ShT} = Val(false), trace_level = TraceMinimal(),
        reltol = nothing, store_trace::Val{StT} = Val(false), maxiters = 1000,
        termination_condition = nothing, kwargs...) where {ShT, StT}
    NonlinearSolve.__test_termination_condition(termination_condition, :LeastSquaresOptim)

    f!, u, resid = NonlinearSolve.__construct_extension_f(prob; alias_u0)
    abstol = NonlinearSolve.DEFAULT_TOLERANCE(abstol, eltype(u))
    reltol = NonlinearSolve.DEFAULT_TOLERANCE(reltol, eltype(u))

    if prob.f.jac === nothing && alg.autodiff isa Symbol
        lsoprob = LSO.LeastSquaresProblem(; x = u, f!, y = resid, alg.autodiff,
            J = prob.f.jac_prototype, output_length = length(resid))
    else
        g! = NonlinearSolve.__construct_extension_jac(prob, alg, u, resid; alg.autodiff)
        lsoprob = LSO.LeastSquaresProblem(;
            x = u, f!, y = resid, g!, J = prob.f.jac_prototype,
            output_length = length(resid))
    end

    allocated_prob = LSO.LeastSquaresProblemAllocated(lsoprob, _lso_solver(alg))

    return LeastSquaresOptimJLCache(prob,
        alg,
        allocated_prob,
        (; x_tol = reltol, f_tol = abstol, g_tol = abstol, iterations = maxiters,
            show_trace = ShT, store_trace = StT, show_every = trace_level.print_frequency))
end

function SciMLBase.solve!(cache::LeastSquaresOptimJLCache)
    res = LSO.optimize!(cache.allocated_prob; cache.kwargs...)
    maxiters = cache.kwargs[:iterations]
    retcode = res.x_converged || res.f_converged || res.g_converged ? ReturnCode.Success :
              (res.iterations ≥ maxiters ? ReturnCode.MaxIters :
               ReturnCode.ConvergenceFailure)
    stats = SciMLBase.NLStats(res.f_calls, res.g_calls, -1, -1, res.iterations)
    return SciMLBase.build_solution(
        cache.prob, cache.alg, res.minimizer, res.ssr / 2; retcode, original = res, stats)
end

end
