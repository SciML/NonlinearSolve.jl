using BracketingNonlinearSolve
include("setup_rootfindingtestsnippet.jl")

prob = IntervalNonlinearProblem(quadratic_f, (1.0, 20.0), 2.0)
prob_lin = IntervalNonlinearProblem(linear_f, (-1.0, 1.0), 0.0)

@testset for alg in (Alefeld(), Bisection(), Brent(), ITP(), Ridder(), ModAB(), nothing)
    sol = solve(prob, alg; abstol = 0.0)
    # Test that solution is to floating point precision
    @test sol.retcode == ReturnCode.FloatingPointLimit
    @test quadratic_f(sol.left, 2.0) < 0
    @test quadratic_f(sol.right, 2.0) > 0
    @test nextfloat(sol.left) == sol.right

    # Solve problem with a root representable with floating point
    sol = solve(prob_lin, alg; abstol = 0.0)
    # Test that solution is exact
    @test sol.retcode == ReturnCode.Success
    @test sol.u == 0.0
    @test sol.left == 0.0
    @test sol.right == 0.0
end
