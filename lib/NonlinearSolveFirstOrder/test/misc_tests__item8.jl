using NonlinearSolveFirstOrder
using NonlinearSolveBase: AbsNormSafeBestTerminationMode, AbsNormSafeTerminationMode,
    AbsNormTerminationMode, AbsTerminationMode
using SciMLBase: NonlinearProblem, NonlinearFunction, ReturnCode, solve
using LinearAlgebra: Diagonal
using Test

const linf = Base.Fix1(maximum, abs)

# Analytic Jacobian, so every call to `f` is a residual evaluation.
sq(u, p) = u .* u .- 2.0
dsq(u, p) = 2 .* Diagonal(u)

function iterates_evaluated(alg, tc; u0 = [1.0], maxiters = 1000)
    seen = Vector{Float64}[]
    f = (u, p) -> (push!(seen, copy(u)); sq(u, p))
    prob = NonlinearProblem(NonlinearFunction(f; jac = dsq), u0)
    kw = tc === nothing ? (;) : (; termination_condition = tc)
    sol = solve(prob, alg; maxiters, kw...)
    return sol, seen
end

@testset "residual is not recomputed at the terminating iterate" begin
    @testset "$(nameof(typeof(alg))) / $tcname" for (tcname, tc) in (
                ("default", nothing),
                ("AbsNormTerminationMode", AbsNormTerminationMode(linf)),
                ("AbsNormSafeTerminationMode", AbsNormSafeTerminationMode(linf)),
                ("AbsNormSafeBestTerminationMode", AbsNormSafeBestTerminationMode(linf)),
                ("AbsTerminationMode", AbsTerminationMode()),
            ),
            alg in (NewtonRaphson(), TrustRegion())

        sol, seen = iterates_evaluated(alg, tc)
        @test sol.retcode == ReturnCode.Success
        @test count(==(sol.u), seen) == 1
        @test sol.resid ≈ sq(sol.u, nothing)
    end
end

@testset "a safe-best rollback still refreshes the residual" begin
    # Stopped short of convergence, so the retained best iterate is not the last one.
    cubic(u, p) = [(u[1] - 1.0)^3 + 1.0e-3 * u[1]]
    prob = NonlinearProblem(cubic, [3.0])
    sol = solve(prob, NewtonRaphson(); maxiters = 4)
    @test sol.resid ≈ cubic(sol.u, nothing)
end
