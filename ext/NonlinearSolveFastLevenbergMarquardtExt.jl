module NonlinearSolveFastLevenbergMarquardtExt

using ArrayInterface, NonlinearSolve, SciMLBase
import ConcreteStructs: @concrete
import FastClosures: @closure
import FastLevenbergMarquardt as FastLM
import StaticArraysCore: SArray

@inline function _fast_lm_solver(::FastLevenbergMarquardtJL{linsolve}, x) where {linsolve}
    if linsolve === :cholesky
        return FastLM.CholeskySolver(ArrayInterface.undefmatrix(x))
    elseif linsolve === :qr
        return FastLM.QRSolver(eltype(x), length(x))
    else
        throw(ArgumentError("Unknown FastLevenbergMarquardt Linear Solver: $linsolve"))
    end
end
@inline _fast_lm_solver(::FastLevenbergMarquardtJL{linsolve}, ::SArray) where {linsolve} = linsolve

function SciMLBase.__solve(prob::Union{NonlinearLeastSquaresProblem, NonlinearProblem},
        alg::FastLevenbergMarquardtJL, args...; alias_u0 = false, abstol = nothing,
        reltol = nothing, maxiters = 1000, termination_condition = nothing, kwargs...)
    NonlinearSolve.__test_termination_condition(termination_condition,
        :FastLevenbergMarquardt)

    fn, u, resid = NonlinearSolve.__construct_extension_f(prob; alias_u0,
        can_handle_oop = Val(prob.u0 isa SArray))
    f = if prob.u0 isa SArray
        @closure (u, p) -> fn(u)
    else
        @closure (du, u, p) -> fn(du, u)
    end
    abstol = NonlinearSolve.DEFAULT_TOLERANCE(abstol, eltype(u))
    reltol = NonlinearSolve.DEFAULT_TOLERANCE(reltol, eltype(u))

    _jac_fn = NonlinearSolve.__construct_extension_jac(prob, alg, u, resid; alg.autodiff,
        can_handle_oop = Val(prob.u0 isa SArray))
    jac_fn = if prob.u0 isa SArray
        @closure (u, p) -> _jac_fn(u)
    else
        @closure (J, u, p) -> _jac_fn(J, u)
    end

    solver_kwargs = (; xtol = reltol, ftol = reltol, gtol = abstol, maxit = maxiters,
        alg.factor, alg.factoraccept, alg.factorreject, alg.minscale, alg.maxscale,
        alg.factorupdate, alg.minfactor, alg.maxfactor)

    if prob.u0 isa SArray
        res, fx, info, iter, nfev, njev = FastLM.lmsolve(f, jac_fn, prob.u0;
            solver_kwargs...)
        LM, solver = nothing, nothing
    else
        J = prob.f.jac_prototype === nothing ? similar(u, length(resid), length(u)) :
            zero(prob.f.jac_prototype)
        solver = _fast_lm_solver(alg, u)
        LM = FastLM.LMWorkspace(u, resid, J)

        res, fx, info, iter, nfev, njev, LM, solver = FastLM.lmsolve!(f, jac_fn, LM;
            solver, solver_kwargs...)
    end

    stats = SciMLBase.NLStats(nfev, njev, -1, -1, iter)
    retcode = info == -1 ? ReturnCode.MaxIters : ReturnCode.Success
    return SciMLBase.build_solution(prob, alg, res, fx; retcode,
        original = (res, fx, info, iter, nfev, njev, LM, solver), stats)
end

end
