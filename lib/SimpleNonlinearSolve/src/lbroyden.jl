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
"""
struct LBroyden{batched, TC <: NLSolveTerminationCondition} <:
       AbstractSimpleNonlinearSolveAlgorithm
    termination_condition::TC
    threshold::Int

    function LBroyden(; batched = false, threshold::Int = 27,
        termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
            abstol = nothing,
            reltol = nothing))
        return new{batched, typeof(termination_condition)}(termination_condition, threshold)
    end
end

@views function SciMLBase.__solve(prob::NonlinearProblem, alg::LBroyden{batched}, args...;
    abstol = nothing, reltol = nothing, maxiters = 1000,
    kwargs...) where {batched}
    tc = alg.termination_condition
    mode = DiffEqBase.get_termination_mode(tc)
    threshold = min(maxiters, alg.threshold)
    x = float(prob.u0)

    batched && @assert ndims(x)==2 "Batched LBroyden only supports 2D arrays"

    if x isa Number
        restore_scalar = true
        x = [x]
        f = u -> prob.f(u[], prob.p)
    else
        f = Base.Fix2(prob.f, prob.p)
        restore_scalar = false
    end

    fₙ = f(x)
    T = eltype(x)

    if SciMLBase.isinplace(prob)
        error("LBroyden currently only supports out-of-place nonlinear problems")
    end

    U, Vᵀ = _init_lbroyden_state(batched, x, threshold)

    atol = abstol !== nothing ? abstol :
           (tc.abstol !== nothing ? tc.abstol :
            real(oneunit(eltype(T))) * (eps(real(one(eltype(T)))))^(4 // 5))
    rtol = reltol !== nothing ? reltol :
           (tc.reltol !== nothing ? tc.reltol : eps(real(one(eltype(T))))^(4 // 5))

    if mode ∈ DiffEqBase.SAFE_BEST_TERMINATION_MODES
        error("LBroyden currently doesn't support SAFE_BEST termination modes")
    end

    storage = mode ∈ DiffEqBase.SAFE_TERMINATION_MODES ? NLSolveSafeTerminationResult() :
              nothing
    termination_condition = tc(storage)

    xₙ = x
    xₙ₋₁ = x
    fₙ₋₁ = fₙ
    update = fₙ
    for i in 1:maxiters
        xₙ = xₙ₋₁ .+ update
        fₙ = f(xₙ)
        Δxₙ = xₙ .- xₙ₋₁
        Δfₙ = fₙ .- fₙ₋₁

        if termination_condition(restore_scalar ? [fₙ] : fₙ, xₙ, xₙ₋₁, atol, rtol)
            xₙ = restore_scalar ? xₙ[] : xₙ
            return SciMLBase.build_solution(prob, alg, xₙ, fₙ; retcode = ReturnCode.Success)
        end

        _U = selectdim(U, 1, 1:min(threshold, i))
        _Vᵀ = selectdim(Vᵀ, 2, 1:min(threshold, i))

        vᵀ = _rmatvec(_U, _Vᵀ, Δxₙ)
        mvec = _matvec(_U, _Vᵀ, Δfₙ)
        u = (Δxₙ .- mvec) ./ (sum(vᵀ .* Δfₙ) .+ convert(T, 1e-5))

        selectdim(Vᵀ, 2, mod1(i, threshold)) .= vᵀ
        selectdim(U, 1, mod1(i, threshold)) .= u

        update = -_matvec(selectdim(U, 1, 1:min(threshold, i + 1)),
            selectdim(Vᵀ, 2, 1:min(threshold, i + 1)), fₙ)

        xₙ₋₁ = xₙ
        fₙ₋₁ = fₙ
    end

    xₙ = restore_scalar ? xₙ[] : xₙ
    return SciMLBase.build_solution(prob, alg, xₙ, fₙ; retcode = ReturnCode.MaxIters)
end

function _init_lbroyden_state(batched::Bool, x, threshold)
    T = eltype(x)
    if batched
        U = fill!(similar(x, (threshold, size(x, 1), size(x, 2))), zero(T))
        Vᵀ = fill!(similar(x, (size(x, 1), threshold, size(x, 2))), zero(T))
    else
        U = fill!(similar(x, (threshold, length(x))), zero(T))
        Vᵀ = fill!(similar(x, (length(x), threshold)), zero(T))
    end
    return U, Vᵀ
end

function _rmatvec(U::AbstractMatrix, Vᵀ::AbstractMatrix,
    x::Union{<:AbstractVector, <:Number})
    length(U) == 0 && return x
    return -x .+ vec((x' * Vᵀ) * U)
end

function _rmatvec(U::AbstractArray{T1, 3}, Vᵀ::AbstractArray{T2, 3},
    x::AbstractMatrix) where {T1, T2}
    length(U) == 0 && return x
    Vᵀx = sum(Vᵀ .* reshape(x, size(x, 1), 1, size(x, 2)); dims = 1)
    return -x .+ _drdims_sum(U .* permutedims(Vᵀx, (2, 1, 3)); dims = 1)
end

function _matvec(U::AbstractMatrix, Vᵀ::AbstractMatrix,
    x::Union{<:AbstractVector, <:Number})
    length(U) == 0 && return x
    return -x .+ vec(Vᵀ * (U * x))
end

function _matvec(U::AbstractArray{T1, 3}, Vᵀ::AbstractArray{T2, 3},
    x::AbstractMatrix) where {T1, T2}
    length(U) == 0 && return x
    xUᵀ = sum(reshape(x, size(x, 1), 1, size(x, 2)) .* permutedims(U, (2, 1, 3)); dims = 1)
    return -x .+ _drdims_sum(xUᵀ .* Vᵀ; dims = 2)
end

_drdims_sum(args...; dims = :) = dropdims(sum(args...; dims); dims)
