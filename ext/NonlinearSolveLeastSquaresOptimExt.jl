module NonlinearSolveLeastSquaresOptimExt

using NonlinearSolve, SciMLBase
import ConcreteStructs: @concrete
import LeastSquaresOptim as LSO

@inline function _lso_solver(::LeastSquaresOptimJL{alg, linsolve}) where {alg, linsolve}
    ls = linsolve === :qr ? LSO.QR() :
         (linsolve === :cholesky ? LSO.Cholesky() :
          (linsolve === :lsmr ? LSO.LSMR() : nothing))
    if alg === :lm
        return LSO.LevenbergMarquardt(ls)
    elseif alg === :dogleg
        return LSO.Dogleg(ls)
    else
        throw(ArgumentError("Unknown LeastSquaresOptim Algorithm: $alg"))
    end
end

# TODO: Implement reinit
@concrete struct LeastSquaresOptimJLCache
    prob
    alg
    allocated_prob
    kwargs
end

function SciMLBase.__init(prob::NonlinearLeastSquaresProblem, alg::LeastSquaresOptimJL,
        args...; alias_u0 = false, abstol = nothing, show_trace::Val{ShT} = Val(false),
        trace_level = TraceMinimal(), store_trace::Val{StT} = Val(false), maxiters = 1000,
        reltol = nothing, kwargs...) where {ShT, StT}
    iip = SciMLBase.isinplace(prob)
    u = NonlinearSolve.__maybe_unaliased(prob.u0, alias_u0)

    abstol = NonlinearSolve.DEFAULT_TOLERANCE(abstol, eltype(u))
    reltol = NonlinearSolve.DEFAULT_TOLERANCE(reltol, eltype(u))

    f! = NonlinearSolve.__make_inplace{iip}(prob.f, prob.p)
    g! = NonlinearSolve.__make_inplace{iip}(prob.f.jac, prob.p)

    resid_prototype = prob.f.resid_prototype === nothing ?
                      (!iip ? prob.f(u, prob.p) : zeros(u)) : prob.f.resid_prototype

    lsoprob = LSO.LeastSquaresProblem(; x = u, f!, y = resid_prototype, g!,
        J = prob.f.jac_prototype, alg.autodiff, output_length = length(resid_prototype))
    allocated_prob = LSO.LeastSquaresProblemAllocated(lsoprob, _lso_solver(alg))

    return LeastSquaresOptimJLCache(prob, alg, allocated_prob,
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
    return SciMLBase.build_solution(cache.prob, cache.alg, res.minimizer, res.ssr / 2;
        retcode, original = res, stats)
end

end
