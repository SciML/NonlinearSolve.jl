module SimpleNonlinearSolveNNlibExt

using ArrayInterface, DiffEqBase, LinearAlgebra, NNlib, SimpleNonlinearSolve, SciMLBase
import SimpleNonlinearSolve: _construct_batched_problem_structure,
    _get_storage, _init_𝓙, _result_from_storage, _get_tolerance, @maybeinplace

function __init__()
    SimpleNonlinearSolve.NNlibExtLoaded[] = true
    return
end

# Broyden's method
@views function SciMLBase.__solve(prob::NonlinearProblem,
    alg::BatchedBroyden;
    abstol = nothing,
    reltol = nothing,
    maxiters = 1000,
    kwargs...)
    iip = isinplace(prob)

    u, f, reconstruct = _construct_batched_problem_structure(prob)
    L, N = size(u)

    tc = alg.termination_condition
    mode = DiffEqBase.get_termination_mode(tc)

    storage = _get_storage(mode, u)

    xₙ, xₙ₋₁, δxₙ, δf = ntuple(_ -> copy(u), 4)
    T = eltype(u)

    atol = _get_tolerance(abstol, tc.abstol, T)
    rtol = _get_tolerance(reltol, tc.reltol, T)
    termination_condition = tc(storage)

    𝓙⁻¹ = _init_𝓙(xₙ)  # L × L × N
    𝓙⁻¹f, xᵀ𝓙⁻¹δf, xᵀ𝓙⁻¹ = similar(𝓙⁻¹, L, N), similar(𝓙⁻¹, 1, N), similar(𝓙⁻¹, 1, L, N)

    @maybeinplace iip fₙ₋₁=f(xₙ) u
    iip && (fₙ = copy(fₙ₋₁))
    for n in 1:maxiters
        batched_mul!(reshape(𝓙⁻¹f, L, 1, N), 𝓙⁻¹, reshape(fₙ₋₁, L, 1, N))
        xₙ .= xₙ₋₁ .- 𝓙⁻¹f

        @maybeinplace iip fₙ=f(xₙ)
        δxₙ .= xₙ .- xₙ₋₁
        δf .= fₙ .- fₙ₋₁

        batched_mul!(reshape(𝓙⁻¹f, L, 1, N), 𝓙⁻¹, reshape(δf, L, 1, N))
        δxₙᵀ = reshape(δxₙ, 1, L, N)

        batched_mul!(reshape(xᵀ𝓙⁻¹δf, 1, 1, N), δxₙᵀ, reshape(𝓙⁻¹f, L, 1, N))
        batched_mul!(xᵀ𝓙⁻¹, δxₙᵀ, 𝓙⁻¹)
        δxₙ .= (δxₙ .- 𝓙⁻¹f) ./ (xᵀ𝓙⁻¹δf .+ T(1e-5))
        batched_mul!(𝓙⁻¹, reshape(δxₙ, L, 1, N), xᵀ𝓙⁻¹, one(T), one(T))

        if termination_condition(fₙ, xₙ, xₙ₋₁, atol, rtol)
            retcode, xₙ, fₙ = _result_from_storage(storage, xₙ, fₙ, f, mode, iip)
            return DiffEqBase.build_solution(prob,
                alg,
                reconstruct(xₙ),
                reconstruct(fₙ);
                retcode)
        end

        xₙ₋₁ .= xₙ
        fₙ₋₁ .= fₙ
    end

    if mode ∈ DiffEqBase.SAFE_BEST_TERMINATION_MODES
        xₙ = storage.u
        @maybeinplace iip fₙ=f(xₙ)
    end

    return DiffEqBase.build_solution(prob,
        alg,
        reconstruct(xₙ),
        reconstruct(fₙ);
        retcode = ReturnCode.MaxIters)
end

# Limited Memory Broyden's method
@views function SciMLBase.__solve(prob::NonlinearProblem,
    alg::BatchedLBroyden;
    abstol = nothing,
    reltol = nothing,
    maxiters = 1000,
    kwargs...)
    iip = isinplace(prob)

    u, f, reconstruct = _construct_batched_problem_structure(prob)
    L, N = size(u)
    T = eltype(u)

    tc = alg.termination_condition
    mode = DiffEqBase.get_termination_mode(tc)

    storage = _get_storage(mode, u)

    η = min(maxiters, alg.threshold)
    U = fill!(similar(u, (η, L, N)), zero(T))
    Vᵀ = fill!(similar(u, (L, η, N)), zero(T))

    xₙ, xₙ₋₁, δfₙ = ntuple(_ -> copy(u), 3)

    atol = _get_tolerance(abstol, tc.abstol, T)
    rtol = _get_tolerance(reltol, tc.reltol, T)
    termination_condition = tc(storage)

    @maybeinplace iip fₙ₋₁=f(xₙ) u
    iip && (fₙ = copy(fₙ₋₁))
    δxₙ = -copy(fₙ₋₁)
    ηNx = similar(xₙ, η, N)

    for i in 1:maxiters
        @. xₙ = xₙ₋₁ - δxₙ
        @maybeinplace iip fₙ=f(xₙ)
        @. δxₙ = xₙ - xₙ₋₁
        @. δfₙ = fₙ - fₙ₋₁

        if termination_condition(fₙ, xₙ, xₙ₋₁, atol, rtol)
            retcode, xₙ, fₙ = _result_from_storage(storage, xₙ, fₙ, f, mode, iip)
            return DiffEqBase.build_solution(prob,
                alg,
                reconstruct(xₙ),
                reconstruct(fₙ);
                retcode)
        end

        _L = min(i, η)
        _U = U[1:_L, :, :]
        _Vᵀ = Vᵀ[:, 1:_L, :]

        idx = mod1(i, η)

        if i > 1
            partial_ηNx = ηNx[1:_L, :]

            _ηNx = reshape(partial_ηNx, 1, :, N)
            batched_mul!(_ηNx, reshape(δxₙ, 1, L, N), _Vᵀ)
            batched_mul!(Vᵀ[:, idx:idx, :], _ηNx, _U)
            Vᵀ[:, idx, :] .-= δxₙ

            _ηNx = reshape(partial_ηNx, :, 1, N)
            batched_mul!(_ηNx, _U, reshape(δfₙ, L, 1, N))
            batched_mul!(U[idx:idx, :, :], _Vᵀ, _ηNx)
            U[idx, :, :] .-= δfₙ
        else
            Vᵀ[:, idx, :] .= -δxₙ
            U[idx, :, :] .= -δfₙ
        end

        U[idx, :, :] .= (δxₙ .- U[idx, :, :]) ./
                        (sum(Vᵀ[:, idx, :] .* δfₙ; dims = 1) .+
                         convert(T, 1e-5))

        _L = min(i + 1, η)
        _ηNx = reshape(ηNx[1:_L, :], :, 1, N)
        batched_mul!(_ηNx, U[1:_L, :, :], reshape(δfₙ, L, 1, N))
        batched_mul!(reshape(δxₙ, L, 1, N), Vᵀ[:, 1:_L, :], _ηNx)

        xₙ₋₁ .= xₙ
        fₙ₋₁ .= fₙ
    end

    if mode ∈ DiffEqBase.SAFE_BEST_TERMINATION_MODES
        xₙ = storage.u
        @maybeinplace iip fₙ=f(xₙ)
    end

    return DiffEqBase.build_solution(prob,
        alg,
        reconstruct(xₙ),
        reconstruct(fₙ);
        retcode = ReturnCode.MaxIters)
end

end
