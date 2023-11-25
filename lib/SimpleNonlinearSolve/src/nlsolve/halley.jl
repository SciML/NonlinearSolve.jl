"""
    SimpleHalley(autodiff)
    SimpleHalley(; autodiff = AutoForwardDiff())

A low-overhead implementation of Halley's Method.

!!! note

    As part of the decreased overhead, this method omits some of the higher level error
    catching of the other methods. Thus, to see better error messages, use one of the other
    methods like `NewtonRaphson`

### Keyword Arguments

  - `autodiff`: determines the backend used for the Hessian. Defaults to
    `AutoForwardDiff()`. Valid choices are `AutoForwardDiff()` or `AutoFiniteDiff()`.
"""
@kwdef @concrete struct SimpleHalley <: AbstractNewtonAlgorithm
    autodiff = AutoForwardDiff()
end

function SciMLBase.__solve(prob::NonlinearProblem, alg::SimpleHalley, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000,
        termination_condition = nothing, kwargs...)
    isinplace(prob) &&
        error("SimpleHalley currently only supports out-of-place nonlinear problems")

    x = copy(float(prob.u0))
    fx = _get_fx(prob, x)
    T = eltype(x)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fx, x,
        termination_condition)

    @bb xo = copy(x)

    if setindex_trait(x) === CanSetindex()
        A = similar(x, length(x), length(x))
        Aaᵢ = similar(x, length(x))
        cᵢ = similar(x)
    else
        A = x
        Aaᵢ = x
        cᵢ = x
    end

    for i in 1:maxiters
        # Hessian Computation is unfortunately type unstable
        fx, dfx, d2fx = compute_jacobian_and_hessian(alg.autodiff, prob, fx, x)
        setindex_trait(x) === CannotSetindex() && (A = dfx)

        aᵢ = dfx \ _vec(fx)
        A_ = _vec(A)
        @bb A_ = d2fx × aᵢ
        A = _restructure(A, A_)

        @bb Aaᵢ = A × aᵢ
        @bb A .*= -1
        bᵢ = dfx \ Aaᵢ

        @bb @. cᵢ = (aᵢ * aᵢ) / (-aᵢ + (T(0.5) * bᵢ))

        if i == 1
            if iszero(fx)
                return build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)
            end
        else
            # Termination Checks
            tc_sol = check_termination(tc_cache, fx, x, xo, prob, alg)
            tc_sol !== nothing && return tc_sol
        end

        @bb @. x += cᵢ

        @bb copyto!(xo, x)
    end

    return build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
