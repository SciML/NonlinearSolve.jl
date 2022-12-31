# TODO add docstrings

# TODO check what the supertype should be
# TODO check if this should be defined as in raphson.jl
struct Broyden <: AbstractSimpleNonlinearSolveAlgorithm end

function SciMLBase.solve(prob::NonlinearProblem,
                         alg::Broyden, args...; abstol = nothing,
                         reltol = nothing,
                         maxiters = 1000, kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    xₙ = float(prob.u0)
    T = typeof(xₙ)
    J⁻¹ = Matrix{T}(I, length(xₙ), length(xₙ))

    if SciMLBase.isinplace(prob)
        error("Broyden currently only supports out-of-place nonlinear problems")
    end

    atol = abstol !== nothing ? abstol :
           real(oneunit(eltype(T))) * (eps(real(one(eltype(T)))))^(4 // 5)
    rtol = reltol !== nothing ? reltol : eps(real(one(eltype(T))))^(4 // 5)

    for _ in 1:maxiters
        # TODO check if nameing with heavy use of subscrips is ok
        fₙ = f(xₙ)
        xₙ₊₁ = xₙ + J⁻¹ * fₙ
        Δxₙ = xₙ₊₁ - xₙ
        Δfₙ = f(xₙ₊₁) - fₙ
        J⁻¹ .+= ((Δxₙ .- J⁻¹ * Δfₙ) ./ (Δxₙ' * J⁻¹ * Δfₙ)) * (Δxₙ' * J⁻¹)

        iszero(fₙ) &&
            return SciMLBase.build_solution(prob, alg, xₙ₊₁, fₙ; retcode = ReturnCode.Success)

        if isapprox(xₙ₊₁, xₙ, atol = atol, rtol = rtol)
            return SciMLBase.build_solution(prob, alg, xₙ₊₁, fₙ; retcode = ReturnCode.Success)
        end
        xₙ = xₙ₊₁
    end

    @show xₙ, fₙ
    return SciMLBase.build_solution(prob, alg, xₙ, fₙ; retcode = ReturnCode.MaxIters)
end
