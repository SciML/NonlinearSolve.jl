Base.@kwdef struct SimpleBatchedDFSane{T, F, TC <: NLSolveTerminationCondition} <:
                   AbstractBatchedNonlinearSolveAlgorithm
    σₘᵢₙ::T = 1.0f-10
    σₘₐₓ::T = 1.0f+10
    σ₁::T = 1.0f0
    M::Int = 10
    γ::T = 1.0f-4
    τₘᵢₙ::T = 0.1f0
    τₘₐₓ::T = 0.5f0
    nₑₓₚ::Int = 2
    ηₛ::F = (f₍ₙₒᵣₘ₎₁, n, xₙ, fₙ) -> f₍ₙₒᵣₘ₎₁ ./ n .^ 2
    termination_condition::TC = NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
        abstol = nothing,
        reltol = nothing)
    max_inner_iterations::Int = 1000
end

function SciMLBase.__solve(prob::NonlinearProblem,
    alg::SimpleBatchedDFSane,
    args...;
    abstol = nothing,
    reltol = nothing,
    maxiters = 100,
    kwargs...)
    iip = isinplace(prob)

    u, f, reconstruct = _construct_batched_problem_structure(prob)
    L, N = size(u)
    T = eltype(u)

    tc = alg.termination_condition
    mode = DiffEqBase.get_termination_mode(tc)

    storage = _get_storage(mode, u)

    atol = _get_tolerance(abstol, tc.abstol, T)
    rtol = _get_tolerance(reltol, tc.reltol, T)
    termination_condition = tc(storage)

    σₘᵢₙ, σₘₐₓ, γ, τₘᵢₙ, τₘₐₓ = T(alg.σₘᵢₙ), T(alg.σₘₐₓ), T(alg.γ), T(alg.τₘᵢₙ), T(alg.τₘₐₓ)
    α₁ = one(T)
    α₊, α₋ = similar(u, 1, N), similar(u, 1, N)
    σₙ = fill(T(alg.σ₁), 1, N)
    𝒹 = similar(σₙ, L, N)
    (; M, nₑₓₚ) = alg

    xₙ, xₙ₋₁, f₍ₙₒᵣₘ₎ₙ₋₁, f₍ₙₒᵣₘ₎ₙ = copy(u), copy(u), similar(u, 1, N), similar(u, 1, N)

    function ff!(fₓ, fₙₒᵣₘ, x)
        f(fₓ, x)
        sum!(abs2, fₙₒᵣₘ, fₓ)
        fₙₒᵣₘ .^= (nₑₓₚ / 2)
        return fₓ
    end

    function ff!(fₙₒᵣₘ, x)
        fₓ = f(x)
        sum!(abs2, fₙₒᵣₘ, fₓ)
        fₙₒᵣₘ .^= (nₑₓₚ / 2)
        return fₓ
    end

    @maybeinplace iip fₙ₋₁=ff!(f₍ₙₒᵣₘ₎ₙ₋₁, xₙ) xₙ
    iip && (fₙ = similar(fₙ₋₁))
    ℋ = repeat(f₍ₙₒᵣₘ₎ₙ₋₁, M, 1)
    f̄ = similar(ℋ, 1, N)
    ηₛ = (n, xₙ, fₙ) -> alg.ηₛ(f₍ₙₒᵣₘ₎ₙ₋₁, n, xₙ, fₙ)

    for n in 1:maxiters
        # Spectral parameter range check
        @. σₙ = sign(σₙ) * clamp(abs(σₙ), σₘᵢₙ, σₘₐₓ)

        # Line search direction
        @. 𝒹 = -σₙ * fₙ₋₁

        η = ηₛ(n, xₙ₋₁, fₙ₋₁)
        maximum!(f̄, ℋ)
        fill!(α₊, α₁)
        fill!(α₋, α₁)
        @. xₙ = xₙ₋₁ + α₊ * 𝒹

        @maybeinplace iip fₙ=ff!(f₍ₙₒᵣₘ₎ₙ, xₙ)

        for _ in 1:(alg.max_inner_iterations)
            𝒸 = @. f̄ + η - γ * α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁

            (sum(f₍ₙₒᵣₘ₎ₙ .≤ 𝒸) ≥ N ÷ 2) && break

            @. α₊ = clamp(α₊^2 * f₍ₙₒᵣₘ₎ₙ₋₁ / (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₊ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
                τₘᵢₙ * α₊,
                τₘₐₓ * α₊)
            @. xₙ = xₙ₋₁ - α₋ * 𝒹
            @maybeinplace iip fₙ=ff!(f₍ₙₒᵣₘ₎ₙ, xₙ)

            (sum(f₍ₙₒᵣₘ₎ₙ .≤ 𝒸) ≥ N ÷ 2) && break

            @. α₋ = clamp(α₋^2 * f₍ₙₒᵣₘ₎ₙ₋₁ / (f₍ₙₒᵣₘ₎ₙ + (T(2) * α₋ - T(1)) * f₍ₙₒᵣₘ₎ₙ₋₁),
                τₘᵢₙ * α₋,
                τₘₐₓ * α₋)
            @. xₙ = xₙ₋₁ + α₊ * 𝒹
            @maybeinplace iip fₙ=ff!(f₍ₙₒᵣₘ₎ₙ, xₙ)
        end

        if termination_condition(fₙ, xₙ, xₙ₋₁, atol, rtol)
            retcode, xₙ, fₙ = _result_from_storage(storage, xₙ, fₙ, f, mode, iip)
            return DiffEqBase.build_solution(prob,
                alg,
                reconstruct(xₙ),
                reconstruct(fₙ);
                retcode)
        end

        # Update spectral parameter
        @. xₙ₋₁ = xₙ - xₙ₋₁
        @. fₙ₋₁ = fₙ - fₙ₋₁

        sum!(abs2, α₊, xₙ₋₁)
        sum!(α₋, xₙ₋₁ .* fₙ₋₁)
        σₙ .= α₊ ./ (α₋ .+ T(1e-5))

        # Take step
        @. xₙ₋₁ = xₙ
        @. fₙ₋₁ = fₙ
        @. f₍ₙₒᵣₘ₎ₙ₋₁ = f₍ₙₒᵣₘ₎ₙ

        # Update history
        ℋ[n % M + 1, :] .= view(f₍ₙₒᵣₘ₎ₙ, 1, :)
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
