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

@concrete struct InplaceFunction{iip} <: Function
    f
end

(f::InplaceFunction{true})(fx, x, p) = f.f(fx, x, p)
(f::InplaceFunction{false})(fx, x, p) = (fx .= f.f(x, p))

function SciMLBase.__init(prob::NonlinearLeastSquaresProblem,
        alg::FastLevenbergMarquardtJL, args...; alias_u0 = false, abstol = nothing,
        reltol = nothing, maxiters = 1000, kwargs...)
    iip = SciMLBase.isinplace(prob)
    u = NonlinearSolve.__maybe_unaliased(prob.u0, alias_u0)
    fu = NonlinearSolve.evaluate_f(prob, u)

    f! = InplaceFunction{iip}(prob.f)

    abstol = NonlinearSolve.DEFAULT_TOLERANCE(abstol, eltype(u))
    reltol = NonlinearSolve.DEFAULT_TOLERANCE(reltol, eltype(u))

    if prob.f.jac === nothing
        use_forward_diff = if alg.autodiff === nothing
            ForwardDiff.can_dual(eltype(u))
        else
            alg.autodiff isa AutoForwardDiff
        end
        uf = SciMLBase.JacobianWrapper{iip}(prob.f, prob.p)
        if use_forward_diff
            cache = iip ? ForwardDiff.JacobianConfig(uf, fu, u) :
                    ForwardDiff.JacobianConfig(uf, u)
        else
            cache = FiniteDiff.JacobianCache(u, fu)
        end
        J! = if iip
            if use_forward_diff
                fu_cache = similar(fu)
                function (J, x, p)
                    uf.p = p
                    ForwardDiff.jacobian!(J, uf, fu_cache, x, cache)
                    return J
                end
            else
                function (J, x, p)
                    uf.p = p
                    FiniteDiff.finite_difference_jacobian!(J, uf, x, cache)
                    return J
                end
            end
        else
            if use_forward_diff
                function (J, x, p)
                    uf.p = p
                    ForwardDiff.jacobian!(J, uf, x, cache)
                    return J
                end
            else
                function (J, x, p)
                    uf.p = p
                    J_ = FiniteDiff.finite_difference_jacobian(uf, x, cache)
                    copyto!(J, J_)
                    return J
                end
            end
        end
    else
        J! = InplaceFunction{iip}(prob.f.jac)
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
    retcode = info == 1 ? ReturnCode.Success :
              (info == -1 ? ReturnCode.MaxIters : ReturnCode.Default)
    return SciMLBase.build_solution(cache.prob, cache.alg, res, fx;
        retcode, original = (res, fx, info, iter, nfev, njev, LM, solver), stats)
end

end
