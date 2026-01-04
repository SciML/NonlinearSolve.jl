module NonlinearSolveFastLevenbergMarquardtExt

using FastClosures: @closure

using ArrayInterface: ArrayInterface
using FastLevenbergMarquardt: FastLevenbergMarquardt
using StaticArraysCore: SArray

using NonlinearSolveBase: NonlinearSolveBase
using NonlinearSolve: NonlinearSolve, FastLevenbergMarquardtJL
using SciMLBase: SciMLBase, AbstractNonlinearProblem, ReturnCode

const FastLM = FastLevenbergMarquardt

function SciMLBase.__solve(
        prob::AbstractNonlinearProblem, alg::FastLevenbergMarquardtJL, args...;
        alias_u0 = false, abstol = nothing, reltol = nothing, maxiters = 1000,
        termination_condition = nothing, kwargs...
    )
    NonlinearSolveBase.assert_extension_supported_termination_condition(
        termination_condition, alg
    )

    f_wrapped, u,
        resid = NonlinearSolveBase.construct_extension_function_wrapper(
        prob; alias_u0, can_handle_oop = Val(prob.u0 isa SArray)
    )
    f = if prob.u0 isa SArray
        @closure (u, p) -> f_wrapped(u)
    else
        @closure (du, u, p) -> f_wrapped(du, u)
    end

    abstol = NonlinearSolveBase.get_tolerance(abstol, eltype(u))
    reltol = NonlinearSolveBase.get_tolerance(reltol, eltype(u))

    jac_fn_wrapped = NonlinearSolveBase.construct_extension_jac(
        prob, alg, u, resid; alg.autodiff, can_handle_oop = Val(prob.u0 isa SArray)
    )
    jac_fn = if prob.u0 isa SArray
        @closure (u, p) -> jac_fn_wrapped(u)
    else
        @closure (J, u, p) -> jac_fn_wrapped(J, u)
    end

    solver_kwargs = (;
        xtol = reltol, ftol = reltol, gtol = abstol, maxit = maxiters,
        alg.factor, alg.factoraccept, alg.factorreject, alg.minscale,
        alg.maxscale, alg.factorupdate, alg.minfactor, alg.maxfactor,
    )

    if prob.u0 isa SArray
        res, fx, info, iter, nfev,
            njev = FastLM.lmsolve(
            f, jac_fn, prob.u0; solver_kwargs...
        )
        LM, solver = nothing, nothing
    else
        J = prob.f.jac_prototype === nothing ? similar(u, length(resid), length(u)) :
            zero(prob.f.jac_prototype)

        solver = if alg.linsolve === :cholesky
            FastLM.CholeskySolver(ArrayInterface.undefmatrix(u))
        elseif alg.linsolve === :qr
            FastLM.QRSolver(eltype(u), length(u))
        else
            throw(ArgumentError("Unknown FastLevenbergMarquardt Linear Solver: \
                                 $(Meta.quot(alg.linsolve))"))
        end

        LM = FastLM.LMWorkspace(u, resid, J)

        res, fx, info, iter, nfev, njev,
            LM, solver = FastLM.lmsolve!(
            f, jac_fn, LM; solver, solver_kwargs...
        )
    end

    stats = SciMLBase.NLStats(nfev, njev, -1, -1, iter)
    retcode = info == -1 ? ReturnCode.MaxIters : ReturnCode.Success
    return SciMLBase.build_solution(
        prob, alg, res, fx; retcode,
        original = (res, fx, info, iter, nfev, njev, LM, solver), stats
    )
end

end
