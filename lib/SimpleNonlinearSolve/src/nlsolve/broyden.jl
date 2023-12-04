"""
    SimpleBroyden()

A low-overhead implementation of Broyden. This method is non-allocating on scalar
and static array problems.
"""
struct SimpleBroyden <: AbstractSimpleNonlinearSolveAlgorithm end

function SciMLBase.__solve(prob::NonlinearProblem, alg::SimpleBroyden, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000, alias_u0 = false,
        termination_condition = nothing, kwargs...)
    x = __maybe_unaliased(prob.u0, alias_u0)
    fx = _get_fx(prob, x)

    @bb xo = copy(x)
    @bb δx = copy(x)
    @bb δf = copy(fx)
    @bb fprev = copy(fx)

    J⁻¹ = __init_identity_jacobian(fx, x)
    @bb J⁻¹δf = copy(x)
    @bb xᵀJ⁻¹ = copy(x)
    @bb δJ⁻¹n = copy(x)
    @bb δJ⁻¹ = copy(J⁻¹)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fx, x,
        termination_condition)

    for _ in 1:maxiters
        @bb δx = J⁻¹ × vec(fprev)
        @bb @. x = xo - δx
        fx = __eval_f(prob, fx, x)
        @bb @. δf = fx - fprev

        # Termination Checks
        tc_sol = check_termination(tc_cache, fx, x, xo, prob, alg)
        tc_sol !== nothing && return tc_sol

        @bb J⁻¹δf = J⁻¹ × vec(δf)
        @bb δx .*= -1
        d = dot(δx, J⁻¹δf)
        @bb xᵀJ⁻¹ = transpose(J⁻¹) × vec(δx)

        @bb @. δJ⁻¹n = (δx - J⁻¹δf) / d

        δJ⁻¹n_ = _vec(δJ⁻¹n)
        xᵀJ⁻¹_ = _vec(xᵀJ⁻¹)
        @bb δJ⁻¹ = δJ⁻¹n_ × transpose(xᵀJ⁻¹_)
        @bb J⁻¹ .+= δJ⁻¹

        @bb copyto!(xo, x)
        @bb copyto!(fprev, fx)
    end

    return build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
