using SimpleNonlinearSolve
include("setup_rootfindtestsnippet.jl")

u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

@testset "$(nameof(typeof(alg)))" for alg in (
        SimpleDFSane(),
        SimpleTrustRegion(),
        SimpleHalley(),
        SimpleTrustRegion(; nlsolve_update_rule = Val(true)),
    )
    sol = run_nlsolve_oop(newton_fails, u0, p; solver = alg)
    @test SciMLBase.successful_retcode(sol)
    @test maximum(abs, newton_fails(sol.u, p)) < 1.0e-9
end
