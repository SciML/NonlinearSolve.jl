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

    for _ in 1:maxiters
        any(iszero, J) && (J = Utils.identity_jacobian!!(J))

        @bb @. δx = fprev / J

        @bb @. x = xo - δx
        fx = NLBUtils.evaluate_f!!(prob, fx, x)

        # Termination Checks
        solved, retcode, fx_sol, x_sol = Utils.check_termination(tc_cache, fx, x, xo, prob)
        solved && return SciMLBase.build_solution(prob, alg, x_sol, fx_sol; retcode)

        @bb δx .*= -1
        @bb @. δx² = δx^2 * J^2
        @bb @. J += (fx - fprev - J * δx) / ifelse(iszero(δx²), T(1.0e-5), δx²) * δx * (J^2)

        @bb copyto!(fprev, fx)
        @bb copyto!(xo, x)
    end

    return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
