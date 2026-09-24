using BracketingNonlinearSolve
include("setup_rootfindingtestsnippet.jl")

using ForwardDiff

@testset for alg in (
        Alefeld(), Bisection(), Brent(), Falsi(), ITP(), Muller(), Ridder(), ModAB(), nothing,
    )
    tspan = (1.0, 20.0)

    function g(p)
        probN = IntervalNonlinearProblem{false}(quadratic_f, typeof(p).(tspan), p)
        return solve(probN, alg; abstol = 1.0e-9).left
    end

    @testset for p in 1.1:0.1:100.0
        @test g(p) ≈ sqrt(p) atol = 1.0e-3 rtol = 1.0e-3
        @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p)) atol = 1.0e-3 rtol = 1.0e-3
    end

    t = (p) -> [sqrt(p[2] / p[1])]
    p = [0.9, 50.0]

    function g2(p)
        probN = IntervalNonlinearProblem{false}(quadratic_f2, tspan, p)
        sol = solve(probN, alg; abstol = 1.0e-9)
        return [sol.u]
    end

    @test g2(p) ≈ [sqrt(p[2] / p[1])] atol = 1.0e-3 rtol = 1.0e-3
    @test ForwardDiff.jacobian(g2, p) ≈ ForwardDiff.jacobian(t, p) atol = 1.0e-3 rtol = 1.0e-3

    probB = IntervalNonlinearProblem{false}(quadratic_f, (1.0, 2.0), 2.0)
    sol = solve(probB, alg; abstol = 1.0e-9)
    @test sol.left ≈ sqrt(2.0) atol = 1.0e-3 rtol = 1.0e-3

    if !(alg isa Bisection || alg isa Falsi)
        probB = IntervalNonlinearProblem{false}(quadratic_f, (sqrt(2.0), 10.0), 2.0)
        sol = solve(probB, alg; abstol = 1.0e-9)
        @test sol.left ≈ sqrt(2.0) atol = 1.0e-3 rtol = 1.0e-3

        probB = IntervalNonlinearProblem{false}(quadratic_f, (0.0, sqrt(2.0)), 2.0)
        sol = solve(probB, alg; abstol = 1.0e-9)
        @test sol.left ≈ sqrt(2.0) atol = 1.0e-3 rtol = 1.0e-3
    end
end
