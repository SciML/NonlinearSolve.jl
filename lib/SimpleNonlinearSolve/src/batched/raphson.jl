struct BatchedSimpleNewtonRaphson{CS, AD, FDT, TC <: NLSolveTerminationCondition} <:
       AbstractBatchedNonlinearSolveAlgorithm
    termination_condition::TC
end

alg_autodiff(alg::BatchedSimpleNewtonRaphson{CS, AD, FDT}) where {CS, AD, FDT} = AD
diff_type(alg::BatchedSimpleNewtonRaphson{CS, AD, FDT}) where {CS, AD, FDT} = FDT

function BatchedSimpleNewtonRaphson(; chunk_size = Val{0}(),
    autodiff = Val{true}(),
    diff_type = Val{:forward},
    termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
        abstol = nothing,
        reltol = nothing))
    return BatchedSimpleNewtonRaphson{SciMLBase._unwrap_val(chunk_size),
        SciMLBase._unwrap_val(autodiff),
        SciMLBase._unwrap_val(diff_type), typeof(termination_condition)}(termination_condition)
end

function SciMLBase.__solve(prob::NonlinearProblem, alg::BatchedSimpleNewtonRaphson;
    abstol = nothing, reltol = nothing, maxiters = 1000, kwargs...)
    iip = SciMLBase.isinplace(prob)
    iip &&
        @assert alg_autodiff(alg) "Inplace BatchedSimpleNewtonRaphson currently only supports autodiff."
    u, f, reconstruct = _construct_batched_problem_structure(prob)

    tc = alg.termination_condition
    mode = DiffEqBase.get_termination_mode(tc)

    storage = _get_storage(mode, u)

    xₙ, xₙ₋₁ = copy(u), copy(u)
    T = eltype(u)

    atol = _get_tolerance(abstol, tc.abstol, T)
    rtol = _get_tolerance(reltol, tc.reltol, T)
    termination_condition = tc(storage)

    if iip
        𝓙 = similar(xₙ, length(xₙ), length(xₙ))
        fₙ = similar(xₙ)
        jac_cfg = ForwardDiff.JacobianConfig(f, fₙ, xₙ)
    end

    for i in 1:maxiters
        if iip
            value_derivative!(𝓙, fₙ, f, xₙ, jac_cfg)
        else
            if alg_autodiff(alg)
                fₙ, 𝓙 = value_derivative(f, xₙ)
            else
                fₙ = f(xₙ)
                𝓙 = FiniteDiff.finite_difference_jacobian(f,
                    xₙ,
                    diff_type(alg),
                    eltype(xₙ),
                    fₙ)
            end
        end

        iszero(fₙ) && return DiffEqBase.build_solution(prob,
            alg,
            reconstruct(xₙ),
            reconstruct(fₙ);
            retcode = ReturnCode.Success)

        δx = reshape(𝓙 \ vec(fₙ), size(xₙ))
        xₙ .-= δx

        if termination_condition(fₙ, xₙ, xₙ₋₁, atol, rtol)
            retcode, xₙ, fₙ = _result_from_storage(storage, xₙ, fₙ, f, mode, iip)
            return DiffEqBase.build_solution(prob,
                alg,
                reconstruct(xₙ),
                reconstruct(fₙ);
                retcode)
        end

        xₙ₋₁ .= xₙ
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
