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
@kwdef @concrete struct SimpleNewtonRaphson <: AbstractSimpleNonlinearSolveAlgorithm
    autodiff = nothing
end

"""
    SimpleGaussNewton(autodiff)
    SimpleGaussNewton(; autodiff = nothing)

Alias for [`SimpleNewtonRaphson`](@ref) used for nonlinear least-squares problems. It
uses the same low-overhead Newton implementation and Jacobian backend selection.
"""
const SimpleGaussNewton = SimpleNewtonRaphson

function configure_autodiff(prob, alg::SimpleNewtonRaphson)
    autodiff = something(alg.autodiff, AutoForwardDiff())
    autodiff = SciMLBase.has_jac(prob.f) ? autodiff :
        NonlinearSolveBase.select_jacobian_autodiff(prob, autodiff)
    @set! alg.autodiff = autodiff
    return alg
end

function SciMLBase.__solve(
        prob::Union{ImmutableNonlinearProblem, NonlinearLeastSquaresProblem},
        alg::SimpleNewtonRaphson, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000,
        alias::Union{Nothing, SciMLBase.NonlinearAliasSpecifier} = nothing,
        alias_u0 = false,
        termination_condition = nothing, kwargs...
    )
    # Extract alias_u0: if alias struct provided, use it; otherwise use alias_u0 kwarg
    _alias_u0 = alias === nothing ? alias_u0 : Utils.get_alias_u0(alias, alias_u0)
    autodiff = alg.autodiff
    x = NLBUtils.maybe_unaliased(prob.u0, _alias_u0)
    fx = NLBUtils.evaluate_f(prob, x)

    solved = iszero(fx)

    abstol, reltol,
        tc_cache = NonlinearSolveBase.init_termination_cache(
        prob, abstol, reltol, fx, x, termination_condition, Val(:simple)
    )

    @bb xo = similar(x)
    fx_cache = Utils.should_cache_fx(prob, prob.f) ?
        NLBUtils.safe_similar(fx) : fx
    jac_cache = Utils.prepare_jacobian(prob, autodiff, fx_cache, x)
    J = Utils.compute_jacobian!!(nothing, prob, autodiff, fx_cache, x, jac_cache)

    retcode, iterations, x, fx, xo, J = Utils.init_loop_state(ReturnCode.Default, x, fx, xo, J)
    fx_sol, x_sol = Utils.fresh(fx), Utils.fresh(x)

    ReactantCore.@trace track_numbers = false while (!solved) & (iterations < maxiters)
        @bb copyto!(xo, x)
        xo = Utils.fresh(xo)
        δx = NLBUtils.restructure(x, J \ NLBUtils.safe_vec(fx))
        @bb x .-= δx

        solved, retcode, fx_sol, x_sol = Utils.check_termination(tc_cache, fx, x, xo, prob)
        fx_sol, x_sol = Utils.fresh(fx_sol), Utils.fresh(x_sol)

        fx = NLBUtils.evaluate_f!!(prob, fx, x)
        J = Utils.compute_jacobian!!(J, prob, autodiff, fx_cache, x, jac_cache)
        iterations += 1
    end

    return Utils.simple_solution(prob, alg, x, fx, x_sol, fx_sol, retcode, solved)
end
