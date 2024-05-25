# File for different types of exotic types
@testsetup module NonlinearSolveExoticTypeTests
using NonlinearSolve

fn_iip = NonlinearFunction{true}((du, u, p) -> du .= u .* u .- p)
fn_oop = NonlinearFunction{false}((u, p) -> u .* u .- p)

u0 = BigFloat[1.0, 1.0, 1.0]
prob_iip_bf = NonlinearProblem{true}(fn_iip, u0, BigFloat(2))
prob_oop_bf = NonlinearProblem{false}(fn_oop, u0, BigFloat(2))

export fn_iip, fn_oop, u0, prob_iip_bf, prob_oop_bf
end

@testitem "BigFloat Support" tags=[:misc] setup=[NonlinearSolveExoticTypeTests] begin
    using NonlinearSolve, LinearAlgebra

    for alg in [NewtonRaphson(), Broyden(), Klement(), DFSane(), TrustRegion()]
        sol = solve(prob_oop_bf, alg)
        @test norm(sol.resid, Inf) < 1e-6
        @test SciMLBase.successful_retcode(sol.retcode)

        sol = solve(prob_iip_bf, alg)
        @test norm(sol.resid, Inf) < 1e-6
        @test SciMLBase.successful_retcode(sol.retcode)
    end
end
