# File for different types of exotic types
@testsetup module SimpleNonlinearSolveExoticTypeTests
using SimpleNonlinearSolve

fn_iip = NonlinearFunction{true}((du, u, p) -> du .= u .* u .- p)
fn_oop = NonlinearFunction{false}((u, p) -> u .* u .- p)

u0 = BigFloat[1.0, 1.0, 1.0]
prob_iip_bf = NonlinearProblem{true}(fn_iip, u0, BigFloat(2))
prob_oop_bf = NonlinearProblem{false}(fn_oop, u0, BigFloat(2))

export fn_iip, fn_oop, u0, prob_iip_bf, prob_oop_bf
end

@testitem "BigFloat Support" tags=[:core] setup=[SimpleNonlinearSolveExoticTypeTests] begin
    using SimpleNonlinearSolve, LinearAlgebra

    for alg in [SimpleNewtonRaphson(), SimpleBroyden(), SimpleKlement(), SimpleDFSane(),
        SimpleTrustRegion(), SimpleLimitedMemoryBroyden(; threshold = 2), SimpleHalley()]
        sol = solve(prob_oop_bf, alg)
        @test norm(sol.resid, Inf) < 1e-6
        @test SciMLBase.successful_retcode(sol.retcode)

        alg isa SimpleHalley && continue

        sol = solve(prob_iip_bf, alg)
        @test norm(sol.resid, Inf) < 1e-6
        @test SciMLBase.successful_retcode(sol.retcode)
    end
end
