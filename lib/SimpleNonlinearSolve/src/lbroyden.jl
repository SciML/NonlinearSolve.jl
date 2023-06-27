"""
    LBroyden(; batched = false,
              termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
                                                                  abstol = nothing, reltol = nothing),
              threshold::Int = 27)

A limited memory implementation of Broyden. This method applies the L-BFGS scheme to
Broyden's method.

!!! warn

    This method is not very stable and can diverge even for very simple problems. This has mostly been
    tested for neural networks in DeepEquilibriumNetworks.jl.

!!! note

    To use the `batched` version, remember to load `NNlib`, i.e., `using NNlib` or
    `import NNlib` must be present in your code.
"""
struct LBroyden{batched, TC <: NLSolveTerminationCondition} <:
       AbstractSimpleNonlinearSolveAlgorithm
    termination_condition::TC
    threshold::Int
end

function LBroyden(; batched = false, threshold::Int = 27,
    termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
        abstol = nothing,
        reltol = nothing))
    if batched
        @assert NNlibExtLoaded[] "Please install and load `NNlib.jl` to use batched Broyden."
        return BatchedLBroyden(termination_condition, threshold)
    end
    return LBroyden{true, typeof(termination_condition)}(termination_condition, threshold)
end

@views function SciMLBase.__solve(prob::NonlinearProblem, alg::LBroyden, args...;
    abstol = nothing, reltol = nothing, maxiters = 1000,
    kwargs...)
    if SciMLBase.isinplace(prob)
        error("LBroyden currently only supports out-of-place nonlinear problems")
    end
    tc = alg.termination_condition
    mode = DiffEqBase.get_termination_mode(tc)
    η = min(maxiters, alg.threshold)
    x = float(prob.u0)

    # FIXME: The scalar case currently is very inefficient
    if x isa Number
        restore_scalar = true
        x = [x]
        f = u -> [prob.f(u[], prob.p)]
    else
        f = Base.Fix2(prob.f, prob.p)
        restore_scalar = false
    end

    L = length(x)
    T = eltype(x)

    U = fill!(similar(x, (η, L)), zero(T))
    Vᵀ = fill!(similar(x, (L, η)), zero(T))

    atol = _get_tolerance(abstol, tc.abstol, T)
    rtol = _get_tolerance(reltol, tc.reltol, T)
    storage = _get_storage(mode, x)
    termination_condition = tc(storage)

    xₙ, xₙ₋₁, δfₙ = ntuple(_ -> copy(x), 3)
    fₙ₋₁ = f(x)
    δxₙ = -copy(fₙ₋₁)
    ηNx = similar(xₙ, η)

    for i in 1:maxiters
        @. xₙ = xₙ₋₁ - δxₙ
        fₙ = f(xₙ)
        @. δxₙ = xₙ - xₙ₋₁
        @. δfₙ = fₙ - fₙ₋₁

        if termination_condition(fₙ, xₙ, xₙ₋₁, atol, rtol)
            retcode, xₙ, fₙ = _result_from_storage(storage, xₙ, fₙ, f, mode, Val(false))
            xₙ = restore_scalar ? xₙ[] : xₙ
            fₙ = restore_scalar ? fₙ[] : fₙ
            return DiffEqBase.build_solution(prob, alg, xₙ, fₙ; retcode)
        end

        _L = min(i, η)
        _U = U[1:_L, :]
        _Vᵀ = Vᵀ[:, 1:_L]

        idx = mod1(i, η)

        partial_ηNx = ηNx[1:_L]

        if i > 1
            _ηNx = reshape(partial_ηNx, 1, :)
            mul!(_ηNx, reshape(δxₙ, 1, L), _Vᵀ)
            mul!(Vᵀ[:, idx:idx], _ηNx, _U)
            Vᵀ[:, idx] .-= δxₙ

            _ηNx = reshape(partial_ηNx, :, 1)
            mul!(_ηNx, _U, reshape(δfₙ, L, 1))
            mul!(U[idx:idx, :], _Vᵀ, _ηNx)
            U[idx, :] .-= δfₙ
        else
            Vᵀ[:, idx] .= -δxₙ
            U[idx, :] .= -δfₙ
        end

        U[idx, :] .= (δxₙ .- U[idx, :]) ./
                     (sum(Vᵀ[:, idx] .* δfₙ) .+
                      convert(T, 1e-5))

        _L = min(i + 1, η)
        _ηNx = reshape(ηNx[1:_L], :, 1)
        mul!(_ηNx, U[1:_L, :], reshape(δfₙ, L, 1))
        mul!(reshape(δxₙ, L, 1), Vᵀ[:, 1:_L], _ηNx)

        xₙ₋₁ .= xₙ
        fₙ₋₁ .= fₙ
    end

    if mode ∈ DiffEqBase.SAFE_BEST_TERMINATION_MODES
        xₙ = storage.u
        fₙ = f(xₙ)
    end

    xₙ = restore_scalar ? xₙ[] : xₙ
    fₙ = restore_scalar ? fₙ[] : fₙ
    return DiffEqBase.build_solution(prob, alg, xₙ, fₙ; retcode = ReturnCode.MaxIters)
end
