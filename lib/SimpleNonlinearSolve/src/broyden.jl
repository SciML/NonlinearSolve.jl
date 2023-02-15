"""
    Broyden(; batched = false)

A low-overhead implementation of Broyden. This method is non-allocating on scalar
and static array problems.
"""
struct Broyden{batched} <: AbstractSimpleNonlinearSolveAlgorithm
    Broyden(; batched = false) = new{batched}()
end

function SciMLBase.__solve(prob::NonlinearProblem, alg::Broyden{false}, args...;
                           abstol = nothing, reltol = nothing, maxiters = 1000, kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)

    fₙ = f(x)
    T = eltype(x)
    J⁻¹ = init_J(x)

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
        xₙ = xₙ₋₁ - J⁻¹ * fₙ₋₁
        fₙ = f(xₙ)
        Δxₙ = xₙ - xₙ₋₁
        Δfₙ = fₙ - fₙ₋₁
        J⁻¹Δfₙ = J⁻¹ * Δfₙ
        J⁻¹ += ((Δxₙ .- J⁻¹Δfₙ) ./ (Δxₙ' * J⁻¹Δfₙ)) * (Δxₙ' * J⁻¹)

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
