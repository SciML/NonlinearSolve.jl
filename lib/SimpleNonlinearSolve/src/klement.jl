"""
    SimpleKlement()

A low-overhead implementation of [Klement](https://jatm.com.br/jatm/article/view/373).
"""
struct SimpleKlement <: AbstractSimpleNonlinearSolveAlgorithm end

function SciMLBase.__solve(prob::NonlinearProblem, alg::SimpleKlement, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000,
        termination_condition = nothing, kwargs...)
    f = isinplace(prob) ? (du, u) -> prob.f(du, u, prob.p) : u -> prob.f(u, prob.p)
    x = float(prob.u0)
    T = eltype(x)
    fx = _get_fx(prob, x)

    singular_tol = eps(T)^(2 // 3)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fx, x,
        termination_condition)

    δx, fprev, xo, δf, d = __copy(fx), __copy(fx), __copy(x), __copy(fx), __copy(x)
    J = __init_identity_jacobian(fx, x)
    J_cache, δx² = __copy(J), __copy(x)

    for _ in 1:maxiters
        if x isa Number
            J < singular_tol && (J = __init_identity_jacobian!!(J))
            F = J
        else
            F = lu(J; check = false)

            # Singularity test
            if any(x -> abs(x) < singular_tol, @view(F.U[diagind(F.U)]))
                J = __init_identity_jacobian!!(J)
                F = lu(J; check = false)
            end
        end

        δx = __copyto!!(δx, fprev)
        δx = __ldiv!!(F, δx)
        x = __sub!!(x, xo, δx)
        fx = __eval_f(prob, f, fx, x)

        # Termination Checks
        tc_sol = check_termination(tc_cache, fx, x, xo, prob, alg)
        tc_sol !== nothing && return tc_sol

        δx = __neg!!(δx)
        δf = __sub!!(δf, fx, fprev)

        # Prevent division by 0
        δx² = __broadcast!!(δx², abs2, δx)
        J_cache = __broadcast!!(J_cache, abs2, J)
        d = _restructure(d, __mul!!(_vec(d), J_cache', _vec(δx²)))
        d = __broadcast!!(d, Base.Fix2(max, singular_tol), d)

        δx² = _restructure(δx², __mul!!(_vec(δx²), J, _vec(δx)))
        δf = __sub!!(δf, δx²)
        δf = __broadcast!!(δf, /, δf, d)

        J_cache = __mul!!(J_cache, _vec(δf), _vec(δx)')
        J_cache = __broadcast!!(J_cache, *, J_cache, J)
        J_cache = __mul!!(J_cache, J_cache, J)

        J = __add!!(J, J_cache)
    end

    return build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
