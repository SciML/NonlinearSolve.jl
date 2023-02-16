module SimpleBatchedNonlinearSolveExt

using ArrayInterfaceCore, LinearAlgebra, SimpleNonlinearSolve, SciMLBase
isdefined(Base, :get_extension) ? (using NNlib) : (using ..NNlib)

_batch_transpose(x) = reshape(x, 1, size(x)...)

_batched_mul(x, y) = x * y

function _batched_mul(x::AbstractArray{T, 3}, y::AbstractMatrix) where {T}
    return dropdims(batched_mul(x, reshape(y, size(y, 1), 1, size(y, 2))); dims = 2)
end

function _batched_mul(x::AbstractMatrix, y::AbstractArray{T, 3}) where {T}
    return batched_mul(reshape(x, size(x, 1), 1, size(x, 2)), y)
end

function _batched_mul(x::AbstractArray{T1, 3}, y::AbstractArray{T2, 3}) where {T1, T2}
    return batched_mul(x, y)
end

function _init_J_batched(x::AbstractMatrix{T}) where {T}
    J = ArrayInterfaceCore.zeromatrix(x[:, 1])
    if ismutable(x)
        J[diagind(J)] .= one(eltype(x))
    else
        J += I
    end
    return repeat(J, 1, 1, size(x, 2))
end

function SciMLBase.__solve(prob::NonlinearProblem, alg::Broyden{true}, args...;
                           abstol = nothing, reltol = nothing, maxiters = 1000, kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)

    if ndims(x) != 2
        error("`batch` mode works only if `ndims(prob.u0) == 2`")
    end

    fₙ = f(x)
    T = eltype(x)
    J⁻¹ = _init_J_batched(x)

    if SciMLBase.isinplace(prob)
        error("Broyden currently only supports out-of-place nonlinear problems")
    end

    atol = abstol !== nothing ? abstol :
           real(oneunit(eltype(T))) * (eps(real(one(eltype(T)))))^(4 // 5)
    rtol = reltol !== nothing ? reltol : eps(real(one(eltype(T))))^(4 // 5)

    xₙ = x
    xₙ₋₁ = x
    fₙ₋₁ = fₙ
    for i in 1:maxiters
        xₙ = xₙ₋₁ .- _batched_mul(J⁻¹, fₙ₋₁)
        fₙ = f(xₙ)
        Δxₙ = xₙ .- xₙ₋₁
        Δfₙ = fₙ .- fₙ₋₁
        J⁻¹Δfₙ = _batched_mul(J⁻¹, Δfₙ)
        J⁻¹ += _batched_mul(((Δxₙ .- J⁻¹Δfₙ) ./
                             (_batched_mul(_batch_transpose(Δxₙ), J⁻¹Δfₙ) .+ T(1e-5))),
                            _batched_mul(_batch_transpose(Δxₙ), J⁻¹))

        iszero(fₙ) &&
            return SciMLBase.build_solution(prob, alg, xₙ, fₙ;
                                            retcode = ReturnCode.Success)

        if isapprox(xₙ, xₙ₋₁, atol = atol, rtol = rtol)
            return SciMLBase.build_solution(prob, alg, xₙ, fₙ;
                                            retcode = ReturnCode.Success)
        end
        xₙ₋₁ = xₙ
        fₙ₋₁ = fₙ
    end

    return SciMLBase.build_solution(prob, alg, xₙ, fₙ; retcode = ReturnCode.MaxIters)
end

end
