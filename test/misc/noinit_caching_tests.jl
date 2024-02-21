@testitem "NoInit Caching" begin
    using LinearAlgebra
    import NLsolve, NLSolvers

    solvers = [SimpleNewtonRaphson(), SimpleTrustRegion(), SimpleDFSane(), NLsolveJL(),
        NLSolversJL(NLSolvers.LineSearch(NLSolvers.Newton(), NLSolvers.Backtracking()))]

    prob = NonlinearProblem((u, p) -> u .^ 2 .- p, [0.1, 0.3], 2.0)

    for alg in solvers
        cache = init(prob, alg)
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid, Inf) ≤ 1e-6

        reinit!(cache; p = 5.0)
        @test cache.prob.p == 5.0
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid, Inf) ≤ 1e-6
        @test norm(sol.u .^ 2 .- 5.0, Inf) ≤ 1e-6
    end
end
