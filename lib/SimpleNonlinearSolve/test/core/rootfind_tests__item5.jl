using SimpleNonlinearSolve
include("setup_rootfindtestsnippet.jl")

prob = NonlinearProblem(quadratic_f, ones(4), 2.0; maxiters = 2)
sol = solve(prob, SimpleNewtonRaphson())
@test sol.retcode === ReturnCode.MaxIters

@testset "Initial state and parameter overrides" begin
    for make_prob in (identity, prob -> convert(SciMLBase.ImmutableNonlinearProblem, prob))
        prob = make_prob(NonlinearProblem{false}((u, p) -> u .* u .- p, 1.0, 1.0))
        for (u0, p, expected) in (
                (nothing, nothing, 1.0), (-1.0, nothing, -1.0),
                (nothing, 4.0, 2.0), (-1.0, 4.0, -2.0),
            )
            sol = solve(prob, SimpleNewtonRaphson(); u0, p)
            @test sol.u ≈ expected
        end
    end
end
