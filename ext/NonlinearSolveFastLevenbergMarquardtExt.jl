module NonlinearSolveFastLevenbergMarquardtExt

using ArrayInterface, NonlinearSolve, SciMLBase
import ConcreteStructs: @concrete
import FastLevenbergMarquardt as FastLM

function _fast_lm_solver(::FastLevenbergMarquardtJL{linsolve}, x) where {linsolve}
    if linsolve == :cholesky
        return FastLM.CholeskySolver(ArrayInterface.undefmatrix(x))
    elseif linsolve == :qr
        return FastLM.QRSolver(eltype(x), length(x))
    else
        throw(ArgumentError("Unknown FastLevenbergMarquardt Linear Solver: $linsolve"))
    end
end

@concrete struct FastLevenbergMarquardtJLCache
    f!
    J!
    prob
    alg
    lmworkspace
    solver
    kwargs
end

@concrete struct InplaceFunction{iip} <: Function
    f
end

(f::InplaceFunction{true})(fx, x, p) = f.f(fx, x, p)
(f::InplaceFunction{false})(fx, x, p) = (fx .= f.f(x, p))

function SciMLBase.__init(prob::NonlinearLeastSquaresProblem,
        alg::FastLevenbergMarquardtJL, args...; abstol = 1e-8, reltol = 1e-8,
        verbose = false, maxiters = 1000, kwargs...)
    iip = SciMLBase.isinplace(prob)

    @assert prob.f.jac!==nothing "FastLevenbergMarquardt requires a Jacobian!"

    f! = InplaceFunction{iip}(prob.f)
    J! = InplaceFunction{iip}(prob.f.jac)

    resid_prototype = prob.f.resid_prototype === nothing ?
                      (!iip ? prob.f(prob.u0, prob.p) : zeros(prob.u0)) :
                      prob.f.resid_prototype

    J = similar(prob.u0, length(resid_prototype), length(prob.u0))

    solver = _fast_lm_solver(alg, prob.u0)
    LM = FastLM.LMWorkspace(prob.u0, resid_prototype, J)

    return FastLevenbergMarquardtJLCache(f!, J!, prob, alg, LM, solver,
        (; xtol = abstol, ftol = reltol, maxit = maxiters, alg.factor, alg.factoraccept,
            alg.factorreject, alg.minscale, alg.maxscale, alg.factorupdate, alg.minfactor,
            alg.maxfactor, kwargs...))
end

function SciMLBase.solve!(cache::FastLevenbergMarquardtJLCache)
    res, fx, info, iter, nfev, njev, LM, solver = FastLM.lmsolve!(cache.f!, cache.J!,
        cache.lmworkspace, cache.prob.p; cache.solver, cache.kwargs...)
    stats = SciMLBase.NLStats(nfev, njev, -1, -1, iter)
    retcode = info == 1 ? ReturnCode.Success :
              (info == -1 ? ReturnCode.MaxIters : ReturnCode.Default)
    return SciMLBase.build_solution(cache.prob, cache.alg, res, fx;
        retcode, original = (res, fx, info, iter, nfev, njev, LM, solver), stats)
end

end
