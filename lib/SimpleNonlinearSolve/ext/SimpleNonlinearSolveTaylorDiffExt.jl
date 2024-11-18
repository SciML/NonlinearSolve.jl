module SimpleNonlinearSolveTaylorDiffExt
using SimpleNonlinearSolve: SimpleNonlinearSolve, SimpleHouseholder, Utils
using NonlinearSolveBase: NonlinearSolveBase, ImmutableNonlinearProblem,
                          AbstractNonlinearSolveAlgorithm
using MaybeInplace: @bb
using FastClosures: @closure
import SciMLBase
import TaylorDiff

SimpleNonlinearSolve.is_extension_loaded(::Val{:TaylorDiff}) = true

const NLBUtils = NonlinearSolveBase.Utils

@inline function __get_higher_order_derivatives(
        ::SimpleHouseholder{N}, prob, x, fx) where {N}
    vN = Val(N)
    l = map(one, x)
    t = TaylorDiff.make_seed(x, l, vN)

    if SciMLBase.isinplace(prob)
        bundle = similar(fx, TaylorDiff.TaylorScalar{eltype(fx), N})
        prob.f(bundle, t, prob.p)
        map!(TaylorDiff.value, fx, bundle)
    else
        bundle = prob.f(t, prob.p)
        fx = map(TaylorDiff.value, bundle)
    end
    invbundle = inv.(bundle)
    num = N == 1 ? map(TaylorDiff.value, invbundle) :
          TaylorDiff.extract_derivative(invbundle, Val(N - 1))
    den = TaylorDiff.extract_derivative(invbundle, vN)
    return num, den, fx
end

function SciMLBase.__solve(prob::ImmutableNonlinearProblem, alg::SimpleHouseholder{N},
        args...; abstol = nothing, reltol = nothing, maxiters = 1000,
        termination_condition = nothing, alias_u0 = false, kwargs...) where {N}
    length(prob.u0) == 1 ||
        throw(ArgumentError("SimpleHouseholder only supports scalar problems"))
    x = NLBUtils.maybe_unaliased(prob.u0, alias_u0)
    fx = NLBUtils.evaluate_f(prob, x)

    iszero(fx) &&
        return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)

    abstol, reltol, tc_cache = NonlinearSolveBase.init_termination_cache(
        prob, abstol, reltol, fx, x, termination_condition, Val(:simple))

    @bb xo = similar(x)

    for i in 1:maxiters
        @bb copyto!(xo, x)
        num, den, fx = __get_higher_order_derivatives(alg, prob, x, fx)
        @bb x .+= N .* num ./ den
        solved, retcode, fx_sol, x_sol = Utils.check_termination(tc_cache, fx, x, xo, prob)
        solved && return SciMLBase.build_solution(prob, alg, x_sol, fx_sol; retcode)
    end

    return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end

function SimpleNonlinearSolve.evaluate_hvvp_internal(hvvp, prob::ImmutableNonlinearProblem, u, a)
    if SciMLBase.isinplace(prob)
        binary_f = @closure (y, x) -> prob.f(y, x, prob.p)
        TaylorDiff.derivative!(hvvp, binary_f, cache.fu, u, a, Val(2))
    else
        unary_f = Base.Fix2(prob.f, prob.p)
        hvvp = TaylorDiff.derivative(unary_f, u, a, Val(2))
    end
    hvvp
end

end
