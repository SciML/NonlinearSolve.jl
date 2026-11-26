using SimpleNonlinearSolve, SciMLBase, Test

@testset "Automatic maximum radius permits adaptive growth" begin
    for scale in (1.0, 1.0e-6)
        prob = NonlinearProblem((x, p) -> scale * (x - 2), 0.0)
        alg = SimpleTrustRegion(; nlsolve_update_rule = Val(true))
        abstol = 1.0e-12
        sol = solve(prob, alg; abstol, reltol = 0)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ 2.0
        @test abs(sol.resid) ≤ abstol
    end
end

@testset "Explicit maximum caps both alternate growth branches" begin
    prob = NonlinearProblem((x, p) -> x - 10, 0.0)
    for expand_threshold in (0.9, 1.5)
        alg = SimpleTrustRegion(;
            initial_trust_radius = 0.5, max_trust_radius = 0.75,
            nlsolve_update_rule = Val(true), expand_threshold
        )
        sol = solve(prob, alg; maxiters = 2)
        @test sol.u ≈ 1.25
    end
end
