"""
```julia
Broyden()
```

A low-overhead implementation of Broyden. This method is non-allocating on scalar
and static array problems.
"""
struct Broyden <: AbstractSimpleNonlinearSolveAlgorithm end

function SciMLBase.__solve(prob::NonlinearProblem, alg::Broyden, args...; abstol = nothing,
                           reltol = nothing, maxiters = 1000, batch = false, kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)

    if batch && ndims(x) != 2
        error("`batch` mode works only if `ndims(prob.u0) == 2`")
    end

    fₙ = f(x)
    T = eltype(x)
    J⁻¹ = init_J(x; batch)

    if SciMLBase.isinplace(prob)
        error("Broyden currently only supports out-of-place nonlinear problems")
    end

    atol = abstol !== nothing ? abstol :
           real(oneunit(eltype(T))) * (eps(real(one(eltype(T)))))^(4 // 5)
    rtol = reltol !== nothing ? reltol : eps(real(one(eltype(T))))^(4 // 5)

    xₙ = x
    xₙ₋₁ = x
    fₙ₋₁ = fₙ
    for _ in 1:maxiters
        xₙ = xₙ₋₁ .- _batched_mul(J⁻¹, fₙ₋₁, batch)
        fₙ = f(xₙ)
        Δxₙ = xₙ .- xₙ₋₁
        Δfₙ = fₙ .- fₙ₋₁
        J⁻¹Δfₙ = _batched_mul(J⁻¹, Δfₙ, batch)
        J⁻¹ += _batched_mul(((Δxₙ .- J⁻¹Δfₙ, batch) ./
                             (_batched_mul(_batch_transpose(Δxₙ, batch), J⁻¹Δfₙ, batch))),
                            _batched_mul(_batch_transpose(Δxₙ, batch), J⁻¹, batch), batch)

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

function _batch_transpose(x, batch)
    !batch && return x'
    return reshape(x, 1, size(x)...)
end
