module NonlinearSolveFastLevenbergMarquardtExt

using ArrayInterface, NonlinearSolve, SciMLBase
import ConcreteStructs: @concrete
import FastClosures: @closure
import FastLevenbergMarquardt as FastLM
import StaticArraysCore: StaticArray

@inline function _fast_lm_solver(::FastLevenbergMarquardtJL{linsolve}, x) where {linsolve}
    if linsolve === :cholesky
        return FastLM.CholeskySolver(ArrayInterface.undefmatrix(x))
    elseif linsolve === :qr
        return FastLM.QRSolver(eltype(x), length(x))
    else
        throw(ArgumentError("Unknown FastLevenbergMarquardt Linear Solver: $linsolve"))
    end
end

# TODO: Implement reinit
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
        reltol = nothing, maxiters = 1000, termination_condition = nothing, kwargs...)
    NonlinearSolve.__test_termination_condition(termination_condition,
        :FastLevenbergMarquardt)
    if prob.u0 isa StaticArray  # FIXME
        error("FastLevenbergMarquardtJL does not support StaticArrays yet.")
    end

    _f!, u, resid = NonlinearSolve.__construct_extension_f(prob; alias_u0)
    f! = @closure (du, u, p) -> _f!(du, u)
    abstol = NonlinearSolve.DEFAULT_TOLERANCE(abstol, eltype(u))
    reltol = NonlinearSolve.DEFAULT_TOLERANCE(reltol, eltype(u))

    _J! = NonlinearSolve.__construct_extension_jac(prob, alg, u, resid; alg.autodiff)
    J! = @closure (J, u, p) -> _J!(J, u)
    J = prob.f.jac_prototype === nothing ? similar(u, length(resid), length(u)) :
        zero(prob.f.jac_prototype)

    solver = _fast_lm_solver(alg, u)
    LM = FastLM.LMWorkspace(u, resid, J)

    return FastLevenbergMarquardtJLCache(f!, J!, prob, alg, LM, solver,
        (; xtol = reltol, ftol = reltol, gtol = abstol, maxit = maxiters, alg.factor,
            alg.factoraccept, alg.factorreject, alg.minscale, alg.maxscale,
            alg.factorupdate, alg.minfactor, alg.maxfactor))
end

function SciMLBase.solve!(cache::FastLevenbergMarquardtJLCache)
    res, fx, info, iter, nfev, njev, LM, solver = FastLM.lmsolve!(cache.f!, cache.J!,
        cache.lmworkspace; cache.solver, cache.kwargs...)
    stats = SciMLBase.NLStats(nfev, njev, -1, -1, iter)
    retcode = info == -1 ? ReturnCode.MaxIters : ReturnCode.Success
    return SciMLBase.build_solution(cache.prob, cache.alg, res, fx;
        retcode, original = (res, fx, info, iter, nfev, njev, LM, solver), stats)
end

end
