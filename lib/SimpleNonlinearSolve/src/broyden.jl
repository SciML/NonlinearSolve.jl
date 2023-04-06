"""
    Broyden(; batched = false,
            termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
                                                                abstol = nothing, reltol = nothing))

A low-overhead implementation of Broyden. This method is non-allocating on scalar
and static array problems.

!!! note

    To use the `batched` version, remember to load `NNlib`, i.e., `using NNlib` or
    `import NNlib` must be present in your code.
"""
struct Broyden{batched, TC <: NLSolveTerminationCondition} <:
       AbstractSimpleNonlinearSolveAlgorithm
    termination_condition::TC

    function Broyden(; batched = false,
                     termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
                                                                         abstol = nothing,
                                                                         reltol = nothing))
        return new{batched, typeof(termination_condition)}(termination_condition)
    end
end

function SciMLBase.__solve(prob::NonlinearProblem, alg::Broyden{false}, args...;
                           abstol = nothing, reltol = nothing, maxiters = 1000, kwargs...)
    tc = alg.termination_condition
    mode = DiffEqBase.get_termination_mode(tc)
    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)

    fₙ = f(x)
    T = eltype(x)
    J⁻¹ = init_J(x)

    if SciMLBase.isinplace(prob)
        error("Broyden currently only supports out-of-place nonlinear problems")
    end

    atol = abstol !== nothing ? abstol :
           (tc.abstol !== nothing ? tc.abstol :
            real(oneunit(eltype(T))) * (eps(real(one(eltype(T)))))^(4 // 5))
    rtol = reltol !== nothing ? reltol :
           (tc.reltol !== nothing ? tc.reltol : eps(real(one(eltype(T))))^(4 // 5))

    if mode ∈ DiffEqBase.SAFE_BEST_TERMINATION_MODES
        error("Broyden currently doesn't support SAFE_BEST termination modes")
    end

    storage = mode ∈ DiffEqBase.SAFE_TERMINATION_MODES ? NLSolveSafeTerminationResult() :
              nothing
    termination_condition = tc(storage)

    xₙ = x
    xₙ₋₁ = x
    fₙ₋₁ = fₙ
    for _ in 1:maxiters
        xₙ = xₙ₋₁ - J⁻¹ * fₙ₋₁
        fₙ = f(xₙ)
        Δxₙ = xₙ - xₙ₋₁
        Δfₙ = fₙ - fₙ₋₁
        J⁻¹Δfₙ = J⁻¹ * Δfₙ
        J⁻¹ += ((Δxₙ .- J⁻¹Δfₙ) ./ (Δxₙ' * J⁻¹Δfₙ)) * (Δxₙ' * J⁻¹)

        if termination_condition(fₙ, xₙ, xₙ₋₁, atol, rtol)
            return SciMLBase.build_solution(prob, alg, xₙ, fₙ; retcode = ReturnCode.Success)
        end

        xₙ₋₁ = xₙ
        fₙ₋₁ = fₙ
    end

    return SciMLBase.build_solution(prob, alg, xₙ, fₙ; retcode = ReturnCode.MaxIters)
end
