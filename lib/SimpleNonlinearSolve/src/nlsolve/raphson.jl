"""
    SimpleNewtonRaphson(autodiff)
    SimpleNewtonRaphson(; autodiff = nothing)

A low-overhead implementation of Newton-Raphson. This method is non-allocating on scalar
and static array problems.

!!! note

    As part of the decreased overhead, this method omits some of the higher level error
    catching of the other methods. Thus, to see better error messages, use one of the other
    methods like `NewtonRaphson`.

### Keyword Arguments

  - `autodiff`: determines the backend used for the Jacobian. Defaults to  `nothing` (i.e.
    automatic backend selection). Valid choices include jacobian backends from
    `DifferentiationInterface.jl`.
"""
@kwdef @concrete struct SimpleNewtonRaphson <: AbstractNewtonAlgorithm
    autodiff = nothing
end

const SimpleGaussNewton = SimpleNewtonRaphson

function SciMLBase.__solve(
        prob::Union{ImmutableNonlinearProblem, NonlinearLeastSquaresProblem},
        alg::SimpleNewtonRaphson, args...; abstol = nothing, reltol = nothing,
        maxiters = 1000, termination_condition = nothing, alias_u0 = false, kwargs...)
    x = __maybe_unaliased(prob.u0, alias_u0)
    fx = _get_fx(prob, x)
    autodiff = __get_concrete_autodiff(prob, alg.autodiff)
    @bb xo = copy(x)
    f = __fixed_parameter_function(prob)
    J, jac_cache = jacobian_cache(autodiff, prob, f, fx, x)

    abstol, reltol, tc_cache = init_termination_cache(
        prob, abstol, reltol, fx, x, termination_condition)

    for i in 1:maxiters
        fx, dfx = value_and_jacobian(autodiff, prob, f, fx, x, jac_cache; J)

        if i == 1
            iszero(fx) && build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)
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
