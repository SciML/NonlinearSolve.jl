"""
```julia
Klement()
```

A low-overhead implementation of [Klement](https://jatm.com.br/jatm/article/view/373).
This method is non-allocating on scalar problems.
"""
struct Klement <: AbstractSimpleNonlinearSolveAlgorithm end

function SciMLBase.__solve(prob::NonlinearProblem,
    alg::Klement, args...; abstol = nothing,
    reltol = nothing,
    maxiters = 1000, kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)
    fₙ = f(x)
    T = eltype(x)
    singular_tol = 1e-9

    if SciMLBase.isinplace(prob)
        error("Klement currently only supports out-of-place nonlinear problems")
    end

    atol = abstol !== nothing ? abstol :
           real(oneunit(eltype(T))) * (eps(real(one(eltype(T)))))^(4 // 5)
    rtol = reltol !== nothing ? reltol : eps(real(one(eltype(T))))^(4 // 5)

    xₙ = x
    xₙ₋₁ = x
    fₙ₋₁ = fₙ

    # x is scalar
    if x isa Number
        J = 1.0
        for _ in 1:maxiters
            xₙ = xₙ₋₁ - fₙ₋₁ / J
            fₙ = f(xₙ)

            iszero(fₙ) &&
                return SciMLBase.build_solution(prob, alg, xₙ, fₙ;
                    retcode = ReturnCode.Success)

            if isapprox(xₙ, xₙ₋₁, atol = atol, rtol = rtol)
                return SciMLBase.build_solution(prob, alg, xₙ, fₙ;
                    retcode = ReturnCode.Success)
            end

            Δxₙ = xₙ - xₙ₋₁
            Δfₙ = fₙ - fₙ₋₁

            # Prevent division by 0
            denominator = max(J^2 * Δxₙ^2, 1e-9)

            k = (Δfₙ - J * Δxₙ) / denominator
            J += (k * Δxₙ * J) * J

            # Singularity test
            if J < singular_tol
                J = 1.0
            end

            xₙ₋₁ = xₙ
            fₙ₋₁ = fₙ
        end
        # x is a vector
    else
        J = init_J(x)
        for _ in 1:maxiters
            F = lu(J, check = false)

            # Singularity test
            if any(abs.(F.U[diagind(F.U)]) .< singular_tol)
                J = init_J(xₙ)
                F = lu(J, check = false)
            end

            tmp = F \ fₙ₋₁
            xₙ = xₙ₋₁ - tmp
            fₙ = f(xₙ)

            iszero(fₙ) &&
                return SciMLBase.build_solution(prob, alg, xₙ, fₙ;
                    retcode = ReturnCode.Success)

            if isapprox(xₙ, xₙ₋₁, atol = atol, rtol = rtol)
                return SciMLBase.build_solution(prob, alg, xₙ, fₙ;
                    retcode = ReturnCode.Success)
            end

            Δxₙ = xₙ - xₙ₋₁
            Δfₙ = fₙ - fₙ₋₁

            # Prevent division by 0
            denominator = max.(J' .^ 2 * Δxₙ .^ 2, 1e-9)

            k = (Δfₙ - J * Δxₙ) ./ denominator
            J += (k * Δxₙ' .* J) * J

            xₙ₋₁ = xₙ
            fₙ₋₁ = fₙ
        end
    end

    return SciMLBase.build_solution(prob, alg, xₙ, fₙ; retcode = ReturnCode.MaxIters)
end
