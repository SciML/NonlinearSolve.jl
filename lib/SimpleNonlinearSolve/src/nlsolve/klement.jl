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

    singular_tol = eps(T)^(2 // 3)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fx, x,
        termination_condition)

    @bb δx = copy(x)
    @bb fprev = copy(fx)
    @bb xo = copy(x)
    @bb d = copy(x)

    J = __init_identity_jacobian(fx, x)
    @bb J_cache = similar(J)
    @bb δx² = similar(x)
    @bb J_cache2 = similar(J)
    @bb F = copy(J)

    for _ in 1:maxiters
        if x isa Number
            J < singular_tol && (J = __init_identity_jacobian!!(J))
            F_ = J
        else
            @bb copyto!(F, J)
            if setindex_trait(F) === CanSetindex()
                F_ = lu!(F; check = false)
            else
                F_ = lu(F; check = false)
            end

            # Singularity test
            if !issuccess(F_)
                J = __init_identity_jacobian!!(J)
                @bb copyto!(F, J)
                if setindex_trait(J) === CanSetindex()
                    F_ = lu!(F; check = false)
                else
                    F_ = lu(F; check = false)
                end
            end
        end

        @bb copyto!(δx, fprev)
        if setindex_trait(δx) === CanSetindex()
            ldiv!(F_, _vec(δx))
        else
            δx = _restructure(δx, F_ \ _vec(δx))
        end
        @bb @. x = xo - δx
        fx = __eval_f(prob, fx, x)

        # Termination Checks
        tc_sol = check_termination(tc_cache, fx, x, xo, prob, alg)
        tc_sol !== nothing && return tc_sol

        @bb δx .*= -1
        @bb J_cache .= transpose(J) .^ 2
        @bb @. δx² = δx^2
        @bb d = J_cache × vec(δx²)
        @bb δx² = J × vec(δx)
        @bb @. fprev = (fx - fprev - δx²) / ifelse(iszero(d), singular_tol, d)
        @bb J_cache = vec(fprev) × transpose(_vec(δx))
        @bb @. J_cache *= J
        @bb J_cache2 = J_cache × J
        @bb @. J += J_cache2

        @bb copyto!(fprev, fx)
        @bb copyto!(xo, x)
    end

    return build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
