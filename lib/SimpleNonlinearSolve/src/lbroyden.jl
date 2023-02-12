"""
    LBroyden(threshold::Int = 27)

A limited memory implementation of Broyden. This method applies the L-BFGS scheme to
Broyden's method.
"""
Base.@kwdef struct LBroyden <: AbstractSimpleNonlinearSolveAlgorithm
    threshold::Int = 27
end

@views function SciMLBase.__solve(prob::NonlinearProblem, alg::LBroyden, args...;
                                  abstol = nothing, reltol = nothing, maxiters = 1000,
                                  batch = false, kwargs...)
    threshold = min(maxiters, alg.threshold)
    x = float(prob.u0)

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

    U = fill!(similar(x, (threshold, length(x))), zero(T))
    Vᵀ = fill!(similar(x, (length(x), threshold)), zero(T))

    atol = abstol !== nothing ? abstol :
           real(oneunit(eltype(T))) * (eps(real(one(eltype(T)))))^(4 // 5)
    rtol = reltol !== nothing ? reltol : eps(real(one(eltype(T))))^(4 // 5)

    xₙ = x
    xₙ₋₁ = x
    fₙ₋₁ = fₙ
    update = fₙ
    for i in 1:maxiters
        xₙ = xₙ₋₁ .+ update
        fₙ = f(xₙ)
        Δxₙ = xₙ .- xₙ₋₁
        Δfₙ = fₙ .- fₙ₋₁

        if iszero(fₙ)
            xₙ = restore_scalar ? xₙ[] : xₙ
            return SciMLBase.build_solution(prob, alg, xₙ, fₙ; retcode = ReturnCode.Success)
        end

        if isapprox(xₙ, xₙ₋₁; atol, rtol)
            xₙ = restore_scalar ? xₙ[] : xₙ
            return SciMLBase.build_solution(prob, alg, xₙ, fₙ; retcode = ReturnCode.Success)
        end

        _U = U[1:min(threshold, i), :]
        _Vᵀ = Vᵀ[:, 1:min(threshold, i)]

        vᵀ = _rmatvec(_U, _Vᵀ, Δxₙ)
        mvec = _matvec(_U, _Vᵀ, Δfₙ)
        Δxₙ = (Δxₙ .- mvec) ./ (sum(vᵀ .* Δfₙ) .+ convert(T, 1e-5))

        Vᵀ[:, mod1(i, threshold)] .= vᵀ
        U[mod1(i, threshold), :] .= Δxₙ

        update = -_matvec(U[1:min(threshold, i + 1), :], Vᵀ[:, 1:min(threshold, i + 1)], fₙ)

        xₙ₋₁ = xₙ
        fₙ₋₁ = fₙ
    end

    xₙ = restore_scalar ? xₙ[] : xₙ
    return SciMLBase.build_solution(prob, alg, xₙ, fₙ; retcode = ReturnCode.MaxIters)
end

function _rmatvec(U::AbstractMatrix, Vᵀ::AbstractMatrix,
                  x::Union{<:AbstractVector, <:Number})
    return -x .+ dropdims(sum(U .* sum(Vᵀ .* x; dims = 1)'; dims = 1); dims = 1)
end

function _matvec(U::AbstractMatrix, Vᵀ::AbstractMatrix,
                 x::Union{<:AbstractVector, <:Number})
    return -x .+ dropdims(sum(sum(x .* U'; dims = 1) .* Vᵀ; dims = 2); dims = 2)
end
