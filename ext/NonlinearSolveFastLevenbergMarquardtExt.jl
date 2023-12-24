module NonlinearSolveFastLevenbergMarquardtExt

using ArrayInterface, NonlinearSolve, SciMLBase
import ConcreteStructs: @concrete
import FastLevenbergMarquardt as FastLM
import FiniteDiff, ForwardDiff

@inline function _fast_lm_solver(::FastLevenbergMarquardtJL{linsolve}, x) where {linsolve}
    if linsolve === :cholesky
        return FastLM.CholeskySolver(ArrayInterface.undefmatrix(x))
    elseif linsolve === :qr
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

function SciMLBase.__init(prob::NonlinearLeastSquaresProblem,
        alg::FastLevenbergMarquardtJL, args...; alias_u0 = false, abstol = nothing,
        reltol = nothing, maxiters = 1000, kwargs...)
    # FIXME: Support scalar u0
    prob.u0 isa Number &&
        throw(ArgumentError("FastLevenbergMarquardtJL does not support scalar `u0`"))
    iip = SciMLBase.isinplace(prob)
    u = NonlinearSolve.__maybe_unaliased(prob.u0, alias_u0)
    fu = NonlinearSolve.evaluate_f(prob, u)

    f! = NonlinearSolve.__make_inplace{iip}(prob.f, nothing)

    abstol = NonlinearSolve.DEFAULT_TOLERANCE(abstol, eltype(u))
    reltol = NonlinearSolve.DEFAULT_TOLERANCE(reltol, eltype(u))

    if prob.f.jac === nothing
        alg = NonlinearSolve.get_concrete_algorithm(alg, prob)
        J! = NonlinearSolve.__construct_jac(prob, alg, u;
            can_handle_arbitrary_dims = Val(true))
    else
        J! = NonlinearSolve.__make_inplace{iip}(prob.f.jac, nothing)
    end

    J = similar(u, length(fu), length(u))

    solver = _fast_lm_solver(alg, u)
    LM = FastLM.LMWorkspace(u, fu, J)

    return FastLevenbergMarquardtJLCache(f!, J!, prob, alg, LM, solver,
        (; xtol = reltol, ftol = reltol, gtol = abstol, maxit = maxiters, alg.factor,
            alg.factoraccept, alg.factorreject, alg.minscale, alg.maxscale,
            alg.factorupdate, alg.minfactor, alg.maxfactor))
end

function SciMLBase.solve!(cache::FastLevenbergMarquardtJLCache)
    res, fx, info, iter, nfev, njev, LM, solver = FastLM.lmsolve!(cache.f!, cache.J!,
        cache.lmworkspace, cache.prob.p; cache.solver, cache.kwargs...)
    stats = SciMLBase.NLStats(nfev, njev, -1, -1, iter)
    retcode = info == -1 ? ReturnCode.MaxIters : ReturnCode.Success
    return SciMLBase.build_solution(cache.prob, cache.alg, res, fx;
        retcode, original = (res, fx, info, iter, nfev, njev, LM, solver), stats)
end

end
