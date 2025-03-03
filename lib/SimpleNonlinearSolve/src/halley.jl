"""
    SimpleHalley(autodiff)
    SimpleHalley(; autodiff = nothing)

A low-overhead implementation of Halley's Method.

!!! note

    As part of the decreased overhead, this method omits some of the higher level error
    catching of the other methods. Thus, to see better error messages, use one of the other
    methods like `NewtonRaphson`.

### Keyword Arguments

  - `autodiff`: determines the backend used for the Jacobian. Defaults to  `nothing` (i.e.
    automatic backend selection). Valid choices include jacobian backends from
    `DifferentiationInterface.jl`.
"""
@kwdef @concrete struct SimpleHalley <: AbstractSimpleNonlinearSolveAlgorithm
    autodiff = nothing
end

function configure_autodiff(prob, alg::SimpleHalley)
    autodiff = something(alg.autodiff, AutoForwardDiff())
    autodiff = SciMLBase.has_jac(prob.f) ? autodiff :
               NonlinearSolveBase.select_jacobian_autodiff(prob, autodiff)
    @set! alg.autodiff = autodiff
    alg
end

function SciMLBase.__solve(
        prob::ImmutableNonlinearProblem, alg::SimpleHalley, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000,
        alias_u0 = false, termination_condition = nothing, kwargs...
)
    autodiff = alg.autodiff
    x = NLBUtils.maybe_unaliased(prob.u0, alias_u0)
    fx = NLBUtils.evaluate_f(prob, x)
    T = promote_type(eltype(fx), eltype(x))

    iszero(fx) &&
        return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)

    abstol, reltol, tc_cache = NonlinearSolveBase.init_termination_cache(
        prob, abstol, reltol, fx, x, termination_condition, Val(:simple)
    )

    @bb xo = copy(x)

    fx_cache = (SciMLBase.isinplace(prob) && !SciMLBase.has_jac(prob.f)) ?
               NLBUtils.safe_similar(fx) : fx
    jac_cache = Utils.prepare_jacobian(prob, autodiff, fx_cache, x)

    if NLBUtils.can_setindex(x)
        Aaᵢ = NLBUtils.safe_similar(x, length(x))
        cᵢ = NLBUtils.safe_similar(x)
    else
        Aaᵢ, cᵢ = x, x, x
    end

    J = Utils.compute_jacobian!!(nothing, prob, autodiff, fx_cache, x, jac_cache)
    for _ in 1:maxiters
        NLBUtils.can_setindex(x) || (A = J)

        # Factorize Once and Reuse
        J_fact = if J isa Number
            J
        else
            fact = LinearAlgebra.lu(J; check = false)
            !LinearAlgebra.issuccess(fact) && return SciMLBase.build_solution(
                prob, alg, x, fx; retcode = ReturnCode.Unstable
            )
            fact
        end

        aᵢ = J_fact \ NLBUtils.safe_vec(fx)
        hvvp = Utils.compute_hvvp(prob, autodiff, fx_cache, x, aᵢ)
        bᵢ = J_fact \ NLBUtils.safe_vec(hvvp)

        cᵢ_ = NLBUtils.safe_vec(cᵢ)
        @bb @. cᵢ_ = (aᵢ * aᵢ) / (-aᵢ + (T(0.5) * bᵢ))
        cᵢ = NLBUtils.restructure(cᵢ, cᵢ_)

        solved, retcode, fx_sol, x_sol = Utils.check_termination(tc_cache, fx, x, xo, prob)
        solved && return SciMLBase.build_solution(prob, alg, x_sol, fx_sol; retcode)

        @bb @. x += cᵢ
        @bb copyto!(xo, x)

        fx = NLBUtils.evaluate_f!!(prob, fx, x)
        J = Utils.compute_jacobian!!(J, prob, autodiff, fx_cache, x, jac_cache)
    end

    return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
