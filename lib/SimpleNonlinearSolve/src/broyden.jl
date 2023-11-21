"""
    SimpleBroyden()

A low-overhead implementation of Broyden. This method is non-allocating on scalar
and static array problems.
"""
struct SimpleBroyden <: AbstractSimpleNonlinearSolveAlgorithm end

function SciMLBase.__solve(prob::NonlinearProblem, alg::SimpleBroyden, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000,
        termination_condition = nothing, kwargs...)
    f = isinplace(prob) ? (du, u) -> prob.f(du, u, prob.p) : u -> prob.f(u, prob.p)
    x = float(prob.u0)
    fx = _get_fx(prob, x)
    xo, δx, fprev, δf = __copy(x), __copy(x), __copy(fx), __copy(fx)

    J⁻¹ = __init_identity_jacobian(fx, x)
    J⁻¹δf, xᵀJ⁻¹ = __copy(x), __copy(x)
    δJ⁻¹, δJ⁻¹n = __copy(x, J⁻¹), __copy(x)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fx, x,
        termination_condition)

    for _ in 1:maxiters
        δx = _restructure(δx, __mul!!(_vec(δx), J⁻¹, _vec(fprev)))
        x = __sub!!(x, xo, δx)
        fx = __eval_f(prob, f, fx, x)
        δf = __sub!!(δf, fx, fprev)

        # Termination Checks
        tc_sol = check_termination(tc_cache, fx, x, xo, prob, alg)
        tc_sol !== nothing && return tc_sol

        J⁻¹δf = _restructure(J⁻¹δf, __mul!!(_vec(J⁻¹δf), J⁻¹, _vec(δf)))
        d = dot(δx, J⁻¹δf)
        xᵀJ⁻¹ = _restructure(xᵀJ⁻¹, __mul!!(_vec(xᵀJ⁻¹), _vec(δx)', J⁻¹))

        if ArrayInterface.can_setindex(δJ⁻¹n)
            @. δJ⁻¹n = (δx - J⁻¹δf) / d
        else
            δJ⁻¹n = @. (δx - J⁻¹δf) / d
        end

        δJ⁻¹ = __mul!!(δJ⁻¹, δJ⁻¹n, xᵀJ⁻¹')
        J⁻¹ = __add!!(J⁻¹, δJ⁻¹)

        xo = __copyto!!(xo, x)
        fprev = __copyto!!(fprev, fx)
    end

    return build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
