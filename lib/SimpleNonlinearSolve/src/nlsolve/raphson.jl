"""
    SimpleNewtonRaphson(autodiff)
    SimpleNewtonRaphson(; autodiff = AutoForwardDiff())

A low-overhead implementation of Newton-Raphson. This method is non-allocating on scalar
and static array problems.

!!! note

    As part of the decreased overhead, this method omits some of the higher level error
    catching of the other methods. Thus, to see better error messages, use one of the other
    methods like `NewtonRaphson`.

### Keyword Arguments

  - `autodiff`: determines the backend used for the Jacobian. Defaults to
    `AutoForwardDiff()`. Valid choices are `AutoForwardDiff()` or `AutoFiniteDiff()`.
"""
@kwdef @concrete struct SimpleNewtonRaphson <: AbstractNewtonAlgorithm
    autodiff = AutoForwardDiff()
end

const SimpleGaussNewton = SimpleNewtonRaphson

function SciMLBase.__solve(prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem},
        alg::SimpleNewtonRaphson, args...; abstol = nothing, reltol = nothing,
        maxiters = 1000, termination_condition = nothing, kwargs...)
    @bb x = copy(float(prob.u0))
    fx = _get_fx(prob, x)
    @bb xo = copy(x)
    J, jac_cache = jacobian_cache(alg.autodiff, prob.f, fx, x, prob.p)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fx, x,
        termination_condition)

    for i in 1:maxiters
        fx, dfx = value_and_jacobian(alg.autodiff, prob.f, fx, x, prob.p, jac_cache; J)

        if i == 1
            if iszero(fx)
                return build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)
            end
        else
            # Termination Checks
            tc_sol = check_termination(tc_cache, fx, x, xo, prob, alg)
            tc_sol !== nothing && return tc_sol
        end

        @bb copyto!(xo, x)
        δx = _restructure(x, dfx \ _vec(fx))
        @bb x .-= δx
    end

    return build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
