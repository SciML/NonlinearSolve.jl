using BracketingNonlinearSolve

@testset "$(nameof(typeof(alg))) evaluates f only inside the bracket" for alg in (
        Alefeld(), Bisection(), Brent(), Falsi(), ITP(), Ridder(), ModAB(), NewtonBisection(),
        nothing,
    )
    lo, hi = 0.01, 1.0
    outside = Float64[]
    function f(u, p)
        lo ≤ u ≤ hi || push!(outside, u)
        return u * sin(1 / u) - 0.1 - p
    end
    function df(u, p)
        lo ≤ u ≤ hi || push!(outside, u)
        return sin(1 / u) - cos(1 / u) / u
    end
    prob = IntervalNonlinearProblem{false}(IntervalNonlinearFunction{false}(f; jac = df), (lo, hi), 0.01)
    sol = solve(prob, alg)
    @test isempty(outside)
    @test SciMLBase.successful_retcode(sol)
    @test abs(f(sol.u, 0.01)) < 1.0e-10
end
