module NonlinearSolveLeastSquaresOptimExt

using NonlinearSolve, SciMLBase
import ConcreteStructs: @concrete
import LeastSquaresOptim as LSO

function _lso_solver(::LeastSquaresOptimJL{alg, linsolve}) where {alg, linsolve}
    ls = linsolve == :qr ? LSO.QR() :
         (linsolve == :cholesky ? LSO.Cholesky() :
          (linsolve == :lsmr ? LSO.LSMR() : nothing))
    if alg == :lm
        return LSO.LevenbergMarquardt(ls)
    elseif alg == :dogleg
        return LSO.Dogleg(ls)
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

@concrete struct FunctionWrapper{iip}
    f
    p
end

(f::FunctionWrapper{true})(du, u) = f.f(du, u, f.p)
(f::FunctionWrapper{false})(du, u) = (du .= f.f(u, f.p))

function SciMLBase.__init(prob::NonlinearLeastSquaresProblem, alg::LeastSquaresOptimJL,
        args...; abstol = 1e-8, reltol = 1e-8, verbose = false, maxiters = 1000, kwargs...)
    iip = SciMLBase.isinplace(prob)

    f! = FunctionWrapper{iip}(prob.f, prob.p)
    g! = prob.f.jac === nothing ? nothing : FunctionWrapper{iip}(prob.f.jac, prob.p)

    resid_prototype = prob.f.resid_prototype === nothing ?
                      (!iip ? prob.f(prob.u0, prob.p) : zeros(prob.u0)) :
                      prob.f.resid_prototype

    lsoprob = LSO.LeastSquaresProblem(; x = prob.u0, f!, y = resid_prototype, g!,
        J = prob.f.jac_prototype, alg.autodiff, output_length = length(resid_prototype))
    allocated_prob = LSO.LeastSquaresProblemAllocated(lsoprob, _lso_solver(alg))

    return LeastSquaresOptimJLCache(prob, alg, allocated_prob,
        (; x_tol = abstol, f_tol = reltol, iterations = maxiters, show_trace = verbose,
            kwargs...))
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
