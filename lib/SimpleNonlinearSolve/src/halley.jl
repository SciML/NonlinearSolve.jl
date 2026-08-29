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
    autodiff = if alg.autodiff === nothing && ReactantCore.within_compile()
        NonlinearSolveBase.select_jacobian_autodiff(prob, nothing)
    else
        something(alg.autodiff, AutoForwardDiff())
    end
    autodiff = SciMLBase.has_jac(prob.f) ? autodiff :
        NonlinearSolveBase.select_jacobian_autodiff(prob, autodiff)
    @set! alg.autodiff = autodiff
    return alg
end

function SciMLBase.__solve(
        prob::ImmutableNonlinearProblem, alg::SimpleHalley, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000,
        alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = false), termination_condition = nothing, kwargs...
    )
    if haskey(kwargs, :alias_u0)
        alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = kwargs[:alias_u0])
    end
    alias_u0 = alias.alias_u0
    autodiff = alg.autodiff
    x = NLBUtils.maybe_unaliased(prob.u0, alias_u0)
    fx = NLBUtils.evaluate_f(prob, x)
    T = promote_type(eltype(fx), eltype(x))

    solved = iszero(fx)

    abstol, reltol,
        tc_cache = NonlinearSolveBase.init_termination_cache(
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
    retcode, iterations, x, fx, xo, J, cᵢ = Utils.init_loop_state(
        ReturnCode.Default, x, fx, xo, J, cᵢ
    )
    fx_sol, x_sol = Utils.fresh(fx), Utils.fresh(x)
    unstable = false

    ReactantCore.@trace track_numbers = false while (!solved) & (!unstable) &
            (iterations < maxiters)
        NLBUtils.can_setindex(x) || (A = J)

        # Factorize Once and Reuse
        if J isa Number
            J_fact = J
        else
            J_fact = LinearAlgebra.lu(J; check = false)
            unstable = !Utils.factorization_succeeded(J_fact)
        end

        if !unstable
            aᵢ = J_fact \ NLBUtils.safe_vec(fx)
            hvvp = Utils.compute_hvvp(
                prob, autodiff, fx_cache, x, NLBUtils.restructure(x, aᵢ)
            )
            bᵢ = J_fact \ NLBUtils.safe_vec(hvvp)

            cᵢ_ = NLBUtils.safe_vec(cᵢ)
            @bb @. cᵢ_ = (aᵢ * aᵢ) / (-aᵢ + (T(0.5) * bᵢ))
            cᵢ = NLBUtils.restructure(cᵢ, cᵢ_)

            solved, retcode, fx_sol,
                x_sol = Utils.check_termination(tc_cache, fx, x, xo, prob)
            fx_sol, x_sol = Utils.fresh(fx_sol), Utils.fresh(x_sol)

            @bb @. x += ifelse(solved, zero(cᵢ), cᵢ)
            @bb copyto!(xo, x)
            xo = Utils.fresh(xo)

            fx = NLBUtils.evaluate_f!!(prob, fx, x)
            J = Utils.compute_jacobian!!(J, prob, autodiff, fx_cache, x, jac_cache)
        end
        iterations += 1
    end

    unstable && return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Unstable)
    return Utils.simple_solution(prob, alg, x, fx, x_sol, fx_sol, retcode, solved)
end
