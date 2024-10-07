@testitem "BigFloat Support" tags=[:core] begin
    using SimpleNonlinearSolve, LinearAlgebra

    fn_iip = NonlinearFunction{true}((du, u, p) -> du .= u .* u .- p)
    fn_oop = NonlinearFunction{false}((u, p) -> u .* u .- p)

    u0 = BigFloat[1.0, 1.0, 1.0]
    prob_iip_bf = NonlinearProblem{true}(fn_iip, u0, BigFloat(2))
    prob_oop_bf = NonlinearProblem{false}(fn_oop, u0, BigFloat(2))

    @testset "$(nameof(typeof(alg)))" for alg in (
        SimpleNewtonRaphson(),
        SimpleBroyden(),
        SimpleKlement(),
        SimpleDFSane(),
        SimpleTrustRegion(),
        SimpleLimitedMemoryBroyden(),
        SimpleHalley()
    )
        sol = solve(prob_oop_bf, alg)
        @test maximum(abs, sol.resid) < 1e-6
        @test SciMLBase.successful_retcode(sol.retcode)

        alg isa SimpleHalley && continue

        sol = solve(prob_iip_bf, alg)
        @test maximum(abs, sol.resid) < 1e-6
        @test SciMLBase.successful_retcode(sol.retcode)
    end
end
