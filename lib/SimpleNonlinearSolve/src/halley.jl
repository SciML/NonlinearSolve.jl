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

function SciMLBase.__solve(
        prob::ImmutableNonlinearProblem, alg::SimpleHalley, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000,
        alias_u0 = false, termination_condition = nothing, kwargs...)
    x = Utils.maybe_unaliased(prob.u0, alias_u0)
    fx = Utils.get_fx(prob, x)
    fx = Utils.eval_f(prob, fx, x)
    T = promote_type(eltype(fx), eltype(x))

    iszero(fx) &&
        return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)

    abstol, reltol, tc_cache = NonlinearSolveBase.init_termination_cache(
        prob, abstol, reltol, fx, x, termination_condition, Val(:simple))

    autodiff = NonlinearSolveBase.select_jacobian_autodiff(prob, alg.autodiff)

    @bb xo = copy(x)

    strait = setindex_trait(x)

    A = strait isa CanSetindex ? similar(x, length(x), length(x)) : x
    Aaᵢ = strait isa CanSetindex ? similar(x, length(x)) : x
    cᵢ = strait isa CanSetindex ? similar(x) : x

    for _ in 1:maxiters
        fx, J, H = Utils.compute_jacobian_and_hessian(autodiff, prob, fx, x)

        strait isa CannotSetindex && (A = J)

        # Factorize Once and Reuse
        J_fact = if J isa Number
            J
        else
            fact = LinearAlgebra.lu(J; check = false)
            !LinearAlgebra.issuccess(fact) && return SciMLBase.build_solution(
                prob, alg, x, fx; retcode = ReturnCode.Unstable)
            fact
        end

        aᵢ = J_fact \ Utils.safe_vec(fx)
        A_ = Utils.safe_vec(A)
        @bb A_ = H × aᵢ
        A = Utils.restructure(A, A_)

        @bb Aaᵢ = A × aᵢ
        @bb A .*= -1
        bᵢ = J_fact \ Utils.safe_vec(Aaᵢ)

        cᵢ_ = Utils.safe_vec(cᵢ)
        @bb @. cᵢ_ = (aᵢ * aᵢ) / (-aᵢ + (T(0.5) * bᵢ))
        cᵢ = Utils.restructure(cᵢ, cᵢ_)

        solved, retcode, fx_sol, x_sol = Utils.check_termination(tc_cache, fx, x, xo, prob)
        solved && return SciMLBase.build_solution(prob, alg, x_sol, fx_sol; retcode)

        @bb @. x += cᵢ
        @bb copyto!(xo, x)
    end

    return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
