"""
    Broyden(; batched = false,
        termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
            abstol = nothing,
            reltol = nothing))

A low-overhead implementation of Broyden. This method is non-allocating on scalar
and static array problems.

!!! note

    To use the `batched` version, remember to load `NNlib`, i.e., `using NNlib` or
    `import NNlib` must be present in your code.
"""
struct Broyden{TC <: NLSolveTerminationCondition} <:
       AbstractSimpleNonlinearSolveAlgorithm
    termination_condition::TC
end

function Broyden(; batched = false,
        termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
            abstol = nothing,
            reltol = nothing))
    if batched
        @assert NNlibExtLoaded[] "Please install and load `NNlib.jl` to use batched Broyden."
        return BatchedBroyden(termination_condition)
    end
    return Broyden(termination_condition)
end

function SciMLBase.__solve(prob::NonlinearProblem, alg::Broyden, args...;
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

    atol = _get_tolerance(abstol, tc.abstol, T)
    rtol = _get_tolerance(reltol, tc.reltol, T)

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
        xₙ = xₙ₋₁ - _restructure(xₙ₋₁, J⁻¹ * _vec(fₙ₋₁))
        fₙ = f(xₙ)
        Δxₙ = xₙ - xₙ₋₁
        Δfₙ = fₙ - fₙ₋₁
        J⁻¹Δfₙ = _restructure(Δfₙ, J⁻¹ * _vec(Δfₙ))
        J⁻¹ += _restructure(J⁻¹,
            ((_vec(Δxₙ) .- _vec(J⁻¹Δfₙ)) ./ (_vec(Δxₙ)' * _vec(J⁻¹Δfₙ))) *
            (_vec(Δxₙ)' * J⁻¹))

        if termination_condition(fₙ, xₙ, xₙ₋₁, atol, rtol)
            return SciMLBase.build_solution(prob, alg, xₙ, fₙ; retcode = ReturnCode.Success)
        end

        xₙ₋₁ = xₙ
        fₙ₋₁ = fₙ
    end

    return SciMLBase.build_solution(prob, alg, xₙ, fₙ; retcode = ReturnCode.MaxIters)
end
