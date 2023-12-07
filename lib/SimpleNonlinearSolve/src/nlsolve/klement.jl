"""
    SimpleKlement()

A low-overhead implementation of [Klement](https://jatm.com.br/jatm/article/view/373). This
method is non-allocating on scalar and static array problems.
"""
struct SimpleKlement <: AbstractSimpleNonlinearSolveAlgorithm end

function SciMLBase.__solve(prob::NonlinearProblem, alg::SimpleKlement, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000, alias_u0 = false,
        termination_condition = nothing, kwargs...)
    x = __maybe_unaliased(prob.u0, alias_u0)
    T = eltype(x)
    fx = _get_fx(prob, x)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fx, x,
        termination_condition)

    @bb δx = copy(x)
    @bb fprev = copy(fx)
    @bb xo = copy(x)
    @bb d = copy(x)

    J = one.(x)
    @bb δx² = similar(x)

    for _ in 1:maxiters
        any(iszero, J) && (J = __init_identity_jacobian!!(J))

        @bb @. δx = fprev / J

        @bb @. x = xo - δx
        fx = __eval_f(prob, fx, x)

        # Termination Checks
        tc_sol = check_termination(tc_cache, fx, x, xo, prob, alg)
        tc_sol !== nothing && return tc_sol

        @bb δx .*= -1
        @bb @. δx² = δx^2 * J^2
        @bb @. J += (fx - fprev - J * δx) / ifelse(iszero(δx²), T(1e-5), δx²) * δx * (J^2)

        @bb copyto!(fprev, fx)
        @bb copyto!(xo, x)
    end

    return build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
