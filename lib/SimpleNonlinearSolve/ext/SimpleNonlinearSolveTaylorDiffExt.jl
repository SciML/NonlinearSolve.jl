module SimpleNonlinearSolveTaylorDiffExt
using SimpleNonlinearSolve
using SimpleNonlinearSolve: ImmutableNonlinearProblem, ReturnCode, build_solution,
                            check_termination, init_termination_cache
using SimpleNonlinearSolve: __maybe_unaliased, _get_fx, __fixed_parameter_function
using MaybeInplace: @bb
using SciMLBase: isinplace

import TaylorDiff

@inline function __get_higher_order_derivatives(
        ::SimpleHouseholder{N}, prob, f, x, fx) where {N}
    vN = Val(N)
    l = map(one, x)
    t = TaylorDiff.make_seed(x, l, vN)

    if isinplace(prob)
        bundle = similar(fx, TaylorDiff.TaylorScalar{eltype(fx), N})
        f(bundle, t)
        map!(TaylorDiff.primal, fx, bundle)
    else
        bundle = f(t)
        fx = map(TaylorDiff.primal, bundle)
    end
    bundle = inv.(bundle)
    num = TaylorDiff.extract_derivative(bundle, N - 1)
    den = TaylorDiff.extract_derivative(bundle, N)
    return num, den, fx
end

function SciMLBase.__solve(prob::ImmutableNonlinearProblem, alg::SimpleHouseholder{N},
        args...; abstol = nothing, reltol = nothing, maxiters = 1000,
        termination_condition = nothing, alias_u0 = false, kwargs...) where {N}
    x = __maybe_unaliased(prob.u0, alias_u0)
    length(x) == 1 ||
        throw(ArgumentError("SimpleHouseholder only supports scalar problems"))
    fx = _get_fx(prob, x)
    @bb xo = copy(x)
    f = __fixed_parameter_function(prob)

    abstol, reltol, tc_cache = init_termination_cache(
        prob, abstol, reltol, fx, x, termination_condition)

    for i in 1:maxiters
        num, den, fx = __get_higher_order_derivatives(alg, prob, f, x, fx)

        if i == 1
            iszero(fx) && build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)
        else
            # Termination Checks
            tc_sol = check_termination(tc_cache, fx, x, xo, prob, alg)
            tc_sol !== nothing && return tc_sol
        end

        @bb copyto!(xo, x)
        @bb x .+= (N - 1) .* num ./ den
    end

    return build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end

end
