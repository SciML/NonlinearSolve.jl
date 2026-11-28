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
    compiling = ReactantCore.within_compile()
    if !compiling && solved
        return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)
    end

    abstol, reltol,
        tc_cache = NonlinearSolveBase.init_termination_cache(
        prob, abstol, reltol, fx, x, termination_condition, Val(:simple)
    )

    solved, retcode, fx_sol, x_sol = Utils.check_termination(tc_cache, fx, x, x, prob)
    if !compiling && (solved || maxiters ≤ 0)
        return Utils.simple_solution(prob, alg, x, fx, x_sol, fx_sol, retcode, solved)
    end

    @bb xo = similar(x)
    fx_cache = Utils.should_cache_fx(prob, prob.f) ?
        NLBUtils.safe_similar(fx) : fx
    jac_cache = Utils.prepare_jacobian(prob, autodiff, fx_cache, x)
    J = nothing

    retcode, iterations, x, fx, xo = Utils.init_loop_state(retcode, x, fx, xo)
    fx_sol, x_sol = Utils.fresh(fx_sol), Utils.fresh(x_sol)

    ReactantCore.@trace track_numbers = false while (!solved) & (iterations < maxiters)
        @bb copyto!(xo, x)
        xo = Utils.fresh(xo)
        δx = if compiling
            compiled_newton_direction(prob, autodiff, fx_cache, x, jac_cache, fx)
        else
            J = Utils.compute_jacobian!!(J, prob, autodiff, fx_cache, x, jac_cache)
            NLBUtils.restructure(x, J \ NLBUtils.safe_vec(fx))
        end
        if x isa Number
            x -= δx
        else
            @bb x .-= δx
        end
        fx = NLBUtils.evaluate_f!!(prob, fx, x)

        solved, retcode, fx_sol, x_sol = Utils.check_termination(tc_cache, fx, x, xo, prob)
        fx_sol, x_sol = Utils.fresh(fx_sol), Utils.fresh(x_sol)

        iterations += 1
    end

    return Utils.simple_solution(prob, alg, x, fx, x_sol, fx_sol, retcode, solved)
end

function compiled_newton_direction(prob, autodiff, fx_cache, x, jac_cache, fx)
    J = Utils.compute_jacobian!!(nothing, prob, autodiff, fx_cache, x, jac_cache)
    return NLBUtils.restructure(x, J \ NLBUtils.safe_vec(fx))
end
