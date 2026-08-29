"""
    SimpleKlement()

A low-overhead implementation of `Klement` [klement2014using](@citep). This
method is non-allocating on scalar and static array problems.
"""
struct SimpleKlement <: AbstractSimpleNonlinearSolveAlgorithm end

function SciMLBase.__solve(
        prob::ImmutableNonlinearProblem, alg::SimpleKlement, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000,
        alias::Union{Nothing, SciMLBase.NonlinearAliasSpecifier} = nothing,
        alias_u0 = false,
        termination_condition = nothing, kwargs...
    )
    # Extract alias_u0: if alias struct provided, use it; otherwise use alias_u0 kwarg
    _alias_u0 = alias === nothing ? alias_u0 : Utils.get_alias_u0(alias, alias_u0)
    x = NLBUtils.maybe_unaliased(prob.u0, _alias_u0)
    T = eltype(x)
    fx = NLBUtils.evaluate_f(prob, x)
    solved = iszero(fx)

    abstol, reltol,
        tc_cache = NonlinearSolveBase.init_termination_cache(
        prob, abstol, reltol, fx, x, termination_condition, Val(:simple)
    )

    @bb δx = copy(x)
    @bb fprev = copy(fx)
    @bb xo = copy(x)
    @bb d = copy(x)

    J = one.(x)
    @bb δx² = similar(x)

    retcode, iterations, x, δx, fprev, xo, J, δx² = Utils.init_loop_state(
        ReturnCode.Default, x, δx, fprev, xo, J, δx²
    )

    ReactantCore.@trace track_numbers = false while (!solved) & (iterations < maxiters)
        # `J` is the diagonal Jacobian approximation, so resetting it is a broadcast.
        reset_jacobian = any(iszero, J)
        @bb @. J = ifelse(reset_jacobian, one(J), J)

        @bb @. δx = fprev / J

        @bb @. x = xo - δx
        fx = NLBUtils.evaluate_f!!(prob, fx, x)

        # Termination Checks
        solved, retcode, fx, x = Utils.check_termination(tc_cache, fx, x, xo, prob)
        fx, x = Utils.fresh(fx), Utils.fresh(x)

        @bb δx .*= -1
        @bb @. δx² = δx^2 * J^2
        @bb @. J += (fx - fprev - J * δx) / ifelse(iszero(δx²), T(1.0e-5), δx²) * δx * (J^2)

        @bb copyto!(fprev, fx)
        @bb copyto!(xo, x)
        fprev, xo = Utils.fresh(fprev), Utils.fresh(xo)
        iterations += 1
    end

    return Utils.simple_solution(prob, alg, x, fx, x, fx, retcode, solved)
end
