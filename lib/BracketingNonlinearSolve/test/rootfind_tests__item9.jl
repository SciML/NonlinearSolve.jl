using BracketingNonlinearSolve

using ForwardDiff

@testset "Overflow-safe secant and midpoint" begin
    s = 1.7e308 # |f(x1)| + |f(x2)| overflows to Inf
    for (f, tspan, root) in (
            ((x, p) -> p * tanh(x - 0.3), (-3.0, 5.0), 0.3),
            ((x, p) -> p * atan(x - 0.3) * 2 / π, (-10.0, 10.0), 0.3),
        )
        sol = solve(IntervalNonlinearProblem(f, tspan, s), ModAB())
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ root atol = 1.0e-12
    end

    # x1 + x2 overflows to Inf
    sol = solve(IntervalNonlinearProblem((x, p) -> x - p, (1.0e308, 1.7e308), 1.5e308), ModAB())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ 1.5e308
end

@testset "NaN residual inside the bracket" begin
    f = (x, p) -> 0.45 < x < 0.55 ? NaN : x - 0.8
    sol = solve(IntervalNonlinearProblem(f, (0.0, 1.0)), ModAB())
    @test sol.retcode == ReturnCode.Failure
end

@testset "Mixed abscissa and residual types" begin
    # Float32 bracket with Float64 parameters
    sol = solve(IntervalNonlinearProblem((x, p) -> x^2 - p, (0.0f0, 3.0f0), 2.0), ModAB())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ sqrt(2.0) atol = 1.0e-5
    @test sol.left isa typeof(sol.right)

    # Residual type depends on the branch (Int or Float64)
    f = (x, p) -> x < 1 ? -1 : x^2 - 2
    sol = solve(IntervalNonlinearProblem(f, (0.0, 3.0)), ModAB())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ sqrt(2.0)

    # ForwardDiff through a closure-captured Dual with a plain Float64 bracket
    g = θ -> solve(IntervalNonlinearProblem((x, p) -> x^2 - θ, (0.0, 3.0)), ModAB()).u
    @test ForwardDiff.derivative(g, 2.0) ≈ 1 / (2 * sqrt(2.0))
end
