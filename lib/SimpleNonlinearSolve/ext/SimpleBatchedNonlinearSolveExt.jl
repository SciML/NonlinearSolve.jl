module SimpleBatchedNonlinearSolveExt

using ArrayInterface, DiffEqBase, LinearAlgebra, SimpleNonlinearSolve, SciMLBase, NNlib
import SimpleNonlinearSolve: _construct_batched_problem_structure, _get_storage, _init_𝓙, _result_from_storage, _get_tolerance, @maybeinplace

@views function SciMLBase.__solve(prob::NonlinearProblem,
    alg::BatchedBroyden;
    abstol=nothing,
    reltol=nothing,
    maxiters=1000,
    kwargs...)
    iip = isinplace(prob)
    u0 = prob.u0

    u, f, reconstruct = _construct_batched_problem_structure(prob)
    L, N = size(u)

    tc = alg.termination_condition
    mode = DiffEqBase.get_termination_mode(tc)

    storage = _get_storage(mode, u)

    xₙ, xₙ₋₁, δx, δf = ntuple(_ -> copy(u), 4)
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
        δx .= xₙ .- xₙ₋₁
        δf .= fₙ .- fₙ₋₁

        batched_mul!(reshape(𝓙⁻¹f, L, 1, N), 𝓙⁻¹, reshape(δf, L, 1, N))
        δxᵀ = reshape(δx, 1, L, N)

        batched_mul!(reshape(xᵀ𝓙⁻¹δf, 1, 1, N), δxᵀ, reshape(𝓙⁻¹f, L, 1, N))
        batched_mul!(xᵀ𝓙⁻¹, δxᵀ, 𝓙⁻¹)
        δx .= (δx .- 𝓙⁻¹f) ./ (xᵀ𝓙⁻¹δf .+ T(1e-5))
        batched_mul!(𝓙⁻¹, reshape(δx, L, 1, N), xᵀ𝓙⁻¹, one(T), one(T))

        if termination_condition(fₙ, xₙ, xₙ₋₁, atol, rtol)
            retcode, xₙ, fₙ = _result_from_storage(storage, xₙ, fₙ, f, mode)
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
        retcode=ReturnCode.MaxIters)
end

end
