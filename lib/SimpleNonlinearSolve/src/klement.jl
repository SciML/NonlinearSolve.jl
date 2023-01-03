"""
```julia
Klement()
```

A low-overhead implementation of [Klement](https://jatm.com.br/jatm/article/view/373).
This method is non-allocating on scalar and static array problems.
"""
struct Klement <: AbstractSimpleNonlinearSolveAlgorithm end

function SciMLBase.solve(prob::NonlinearProblem,
                         alg::Klement, args...; abstol = nothing,
                         reltol = nothing,
                         maxiters = 1000, kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)
    fₙ = f(x)
    T = eltype(x)
    J = ArrayInterfaceCore.zeromatrix(x) + I

    if SciMLBase.isinplace(prob)
        error("Klement currently only supports out-of-place nonlinear problems")
    end

    atol = abstol !== nothing ? abstol :
           real(oneunit(eltype(T))) * (eps(real(one(eltype(T)))))^(4 // 5)
    rtol = reltol !== nothing ? reltol : eps(real(one(eltype(T))))^(4 // 5)

    xₙ = x
    xₙ₋₁ = x
    fₙ₋₁ = fₙ
    for _ in 1:maxiters
        xₙ = xₙ₋₁ - inv(J) * fₙ₋₁
        fₙ = f(xₙ)
        Δxₙ = xₙ - xₙ₋₁
        Δfₙ = fₙ - fₙ₋₁

        # Prevent division by 0
        denominator = max.(J' .^ 2 * Δxₙ .^ 2, 1e-9)

        k = (Δfₙ - J * Δxₙ) ./ denominator
        J += (k * Δxₙ' .* J) * J

        # Prevent inverting singular matrix
        if det(J) ≈ 0
            J = ArrayInterfaceCore.zeromatrix(x) + I
        end

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
