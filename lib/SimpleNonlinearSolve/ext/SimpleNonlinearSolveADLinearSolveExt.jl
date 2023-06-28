module SimpleNonlinearSolveADLinearSolveExt

using AbstractDifferentiation,
    ArrayInterface, DiffEqBase, LinearAlgebra, LinearSolve,
    SimpleNonlinearSolve, SciMLBase
import SimpleNonlinearSolve: _construct_batched_problem_structure,
    _get_storage, _result_from_storage, _get_tolerance, @maybeinplace

const AD = AbstractDifferentiation

function __init__()
    SimpleNonlinearSolve.ADLinearSolveExtLoaded[] = true
    return
end

function SimpleNonlinearSolve.SimpleBatchedNewtonRaphson(; chunk_size = Val{0}(),
    autodiff = Val{true}(),
    diff_type = Val{:forward},
    termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
        abstol = nothing,
        reltol = nothing))
    # TODO: Use `diff_type`. FiniteDiff.jl is currently not available in AD.jl
    chunksize = SciMLBase._unwrap_val(chunk_size) == 0 ? nothing : chunk_size
    ad = SciMLBase._unwrap_val(autodiff) ?
         AD.ForwardDiffBackend(; chunksize) :
         AD.FiniteDifferencesBackend()
    return SimpleBatchedNewtonRaphson{typeof(ad), Nothing, typeof(termination_condition)}(ad,
        nothing,
        termination_condition)
end

function SciMLBase.__solve(prob::NonlinearProblem,
    alg::SimpleBatchedNewtonRaphson;
    abstol = nothing,
    reltol = nothing,
    maxiters = 1000,
    kwargs...)
    iip = isinplace(prob)
    @assert !iip "SimpleBatchedNewtonRaphson currently only supports out-of-place nonlinear problems."
    u, f, reconstruct = _construct_batched_problem_structure(prob)

    tc = alg.termination_condition
    mode = DiffEqBase.get_termination_mode(tc)

    storage = _get_storage(mode, u)

    xₙ, xₙ₋₁, δx = copy(u), copy(u), copy(u)
    T = eltype(u)

    atol = _get_tolerance(abstol, tc.abstol, T)
    rtol = _get_tolerance(reltol, tc.reltol, T)
    termination_condition = tc(storage)

    for i in 1:maxiters
        fₙ, (𝓙,) = AD.value_and_jacobian(alg.autodiff, f, xₙ)

        iszero(fₙ) && return DiffEqBase.build_solution(prob,
            alg,
            reconstruct(xₙ),
            reconstruct(fₙ);
            retcode = ReturnCode.Success)

        solve(LinearProblem(𝓙, vec(fₙ); u0 = vec(δx)), alg.linsolve; kwargs...)
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
        fₙ = f(xₙ)
    end

    return DiffEqBase.build_solution(prob,
        alg,
        reconstruct(xₙ),
        reconstruct(fₙ);
        retcode = ReturnCode.MaxIters)
end

end
