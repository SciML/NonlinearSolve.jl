using NonlinearSolveFirstOrder, NonlinearSolveBase, SciMLBase

@testset "Standalone FirstOrder bounded least-squares default" begin
    prob = NonlinearLeastSquaresProblem(
        (u, p) -> [u[1] - 2, u[2] - 0.25, 1.0],
        [0.5, 0.5]; lb = 0.0, ub = 1.0
    )
    for cached in (false, true)
        if cached
            cache = init(prob)
            @test SciMLBase.allowsbounds(cache.alg)
            @test all(c -> c.prob.lb == 0.0 && c.prob.ub == 1.0, cache.caches)
            sol = solve!(cache)
        else
            sol = solve(prob)
        end
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [1.0, 0.25]
        @test sol.resid ≈ [-1.0, 0.0, 1.0]
    end
    @test SciMLBase.allowsbounds(NonlinearSolveBase.initialization_alg(prob, nothing))
    @test !SciMLBase.allowsbounds(init(SciMLBase.remake(prob; lb = nothing, ub = nothing)).alg)
end
