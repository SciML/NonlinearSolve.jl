using SCCNonlinearSolve, NonlinearSolve, SciMLBase, Test

# `2 - 2u₁³ = 0` is a nonlinear block, `4u₂ = u₁` a linear block fed by it
function f!(du, u, p, t)
    du[1] = 2 - 2u[1]^3
    return du[2] = u[1] - 4u[2]
end
b = [0.0]
sccprob = SciMLBase.SCCNonlinearProblem(
    (NonlinearProblem((u, p) -> [2 - 2u[1]^3], [0.5]), LinearProblem(fill(4.0, 1, 1), b, b)),
    (Returns(nothing), (p, sols) -> (p[1] = sols[1].u[1]; nothing))
)

@testset "Nonlinear `verbose`/`alias` are not forwarded as-is to linear blocks" begin
    kw = (; verbose = NonlinearVerbosity(), alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = false))
    for alg in (nothing, NewtonRaphson())
        sol = solve(sccprob, alg; kw...)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [1.0, 0.25]
    end
end

# SciML/ModelingToolkit.jl#5194; `lowered_problem` needs SciMLBase >= 3.55
if :lowered_problem in fieldnames(SteadyStateProblem)
    @testset "Default solve of a SteadyStateProblem with an SCC lowering" begin
        prob = SteadyStateProblem(f!, [0.5, 0.0]; lowered_problem = sccprob)
        for sol in (solve(prob), solve(prob, nothing))
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ [1.0, 0.25]
        end
    end
end
