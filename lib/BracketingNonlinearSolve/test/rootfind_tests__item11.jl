using BracketingNonlinearSolve

# L. Tomov's counterexample (f101 of the modAB benchmark set): the root x = 0 is flat on
# both sides and seven times steeper on the right, so |f| ~ x^2 underflows long before
# the bracket closes and secant steps creep towards the root from one side.
tomov(x, _ = nothing) = x < 0 ? -x * x : 7 * x * x

@testset "$(nameof(typeof(alg))) solves Tomov's counterexample" for alg in (
        Alefeld(), Bisection(), Brent(), ITP(), Ridder(), ModAB(), nothing,
    )
    sol = solve(IntervalNonlinearProblem{false}(tomov, (-π, 1.0)), alg; abstol = 1.0e-14)
    @test SciMLBase.successful_retcode(sol)
    @test abs(sol.u) < 2.0e-14
    @test sol.left ≤ 0 ≤ sol.right
end

@testset "ModAB limits AB steps kept by the residual test alone" begin
    # Each AB step here halves the residual while hardly shrinking the bracket, so a
    # fallback test that keeps AB going as long as the residual halves never switches
    # back to bisection: the previous implementation needed 778 evaluations, and 824
    # without the `MaxResidualSteps` cap. With the cap it needs 76.
    n = Ref(0)
    prob = IntervalNonlinearProblem{false}((x, p) -> (n[] += 1; tomov(x)), (-π, 1.0))
    sol = solve(prob, ModAB(); abstol = 1.0e-14)
    @test SciMLBase.successful_retcode(sol)
    @test abs(sol.u) < 2.0e-14
    @test n[] ≤ 100
end
