using BracketingNonlinearSolve, Unitful
import DynamicQuantities as DQ

f(u, p) = u^2 - p
g(u, p) = sin(u)
h(u, p) = exp(-u) * sin(u)
i(u, p) = u^3 - 1

@testset "Quadratic function" begin
    tspan = (10.0, 30.0)
    p = 612.0
    prob = IntervalNonlinearProblem{false}(f, tspan, p)
    sol = solve(prob, Muller())

    @test sol.u ≈ √612

    tspan = (-10.0, -30.0)
    prob = IntervalNonlinearProblem{false}(f, tspan, p)
    sol = solve(prob, Muller())

    @test sol.u ≈ -√612
end

@testset "Sine function" begin
    tspan = (1.0, 3.0)
    prob = IntervalNonlinearProblem{false}(g, tspan)
    sol = solve(prob, Muller())

    @test sol.u ≈ π

    tspan = (2.0, 6.0)
    prob = IntervalNonlinearProblem{false}(g, tspan)
    sol = solve(prob, Muller())

    @test sol.u ≈ 2 * π
end

@testset "Exponential-sine function" begin
    tspan = (-2.0, -4.0)
    prob = IntervalNonlinearProblem{false}(h, tspan)
    sol = solve(prob, Muller())

    @test sol.u ≈ -π

    tspan = (-3.0, 1.0)
    prob = IntervalNonlinearProblem{false}(h, tspan)
    sol = solve(prob, Muller())

    @test sol.u ≈ 0 atol = 1.0e-15

    tspan = (-1.0, 1.0)
    prob = IntervalNonlinearProblem{false}(h, tspan)
    sol = solve(prob, Muller())

    @test sol.u ≈ π
end

@testset "Complex roots" begin
    tspan = (-1.0, 1.0 * im)
    prob = IntervalNonlinearProblem{false}(i, tspan)
    sol = solve(prob, Muller())

    @test sol.u ≈ (-1 + √3 * im) / 2

    tspan = (-1.0, -1.0 * im)
    prob = IntervalNonlinearProblem{false}(i, tspan)
    sol = solve(prob, Muller())

    @test sol.u ≈ (-1 - √3 * im) / 2
end

@testset "Middle" begin
    tspan = (10.0, 30.0)
    p = 612.0
    prob = IntervalNonlinearProblem{false}(f, tspan, p)
    sol = solve(prob, Muller(20.0))

    @test sol.u ≈ √612

    tspan = (1.0, 3.0)
    prob = IntervalNonlinearProblem{false}(g, tspan)
    sol = solve(prob, Muller(2.0))

    @test sol.u ≈ π

    tspan = (-2.0, -4.0)
    prob = IntervalNonlinearProblem{false}(h, tspan)
    sol = solve(prob, Muller(-3.0))

    @test sol.u ≈ -π

    tspan = (-1.0, 1.0 * im)
    prob = IntervalNonlinearProblem{false}(i, tspan)
    sol = solve(prob, Muller(0.0))

    @test sol.u ≈ (-1 + √3 * im) / 2
end

@testset "Residual tolerance in commensurate units" begin
    # The residual stops on `abstol`, which has the units of x. A residual in km must be
    # compared with 1e-8 m, not with 1e-8 km.
    abstol = 1.0e-8u"m"
    sols = map((u"m", u"km")) do res_unit
        f = (x, p) -> uconvert(res_unit, 1.0e-5 * ((x / 1u"m")^3 - 2) * u"m")
        sol = solve(IntervalNonlinearProblem{false}(f, (0.0u"m", 2.0u"m")), Muller(); abstol)
        @test SciMLBase.successful_retcode(sol)
        @test abs(sol.resid) ≤ abstol
        # |f'| ≈ 4.8e-5 at the root, so a residual below 1e-8 m puts x within ~2e-4 m of it
        @test sol.u ≈ cbrt(2) * u"m" atol = 3.0e-4u"m"
        sol
    end
    @test sols[1].u ≈ sols[2].u # The units of the residual do not change the iterates
    # Incommensurate units: the value of `abstol` is used in the units of `f`
    f = (x, p) -> ((x / 1u"m")^3 - 2) * u"N"
    sol = solve(IntervalNonlinearProblem{false}(f, (0.0u"m", 2.0u"m")), Muller(); abstol)
    @test SciMLBase.successful_retcode(sol)
    @test abs(sol.resid) ≤ 1.0e-8u"N"
end

@testset "Residual tolerance keeps its precision" begin
    # 1e5 overflows Float16: rounded to the type of the residual, the tolerance would be Inf
    # and accept the infinite residual reached after one step
    f = (x, _) -> Float16(1 + x / 1024 + x * (x - 1) * (x - 2))
    sol = solve(IntervalNonlinearProblem{false}(f, (0.0, 2.0)), Muller(); abstol = 1.0e5, maxiters = 1)
    @test !isfinite(sol.resid)
    @test !SciMLBase.successful_retcode(sol)

    # Exact integer residuals at the integral starting samples: 1e-8 is not an Int
    g = (x, _) -> isinteger(x) ? Int(x)^3 - 2 : x^3 - 2
    sol = solve(IntervalNonlinearProblem{false}(g, (0.0, 2.0)), Muller(); abstol = 1.0e-8)
    @test SciMLBase.successful_retcode(sol)
    @test abs(sol.resid) ≤ 1.0e-8
end

@testset "DynamicQuantities residual with another precision or none" begin
    # Float64 bracket and tolerance in s; the residual is a Float32 force or a plain Float64
    tspan = (0.0 * DQ.u"s", 2.0 * DQ.u"s")
    abstol = 1.0e-8 * DQ.u"s"
    force = (x, _) -> Float32(DQ.ustrip(x)^3 - 2) * DQ.u"N"
    plain = (x, _) -> DQ.ustrip(x)^3 - 2
    for (f, tol) in ((force, 1.0e-8 * DQ.u"N"), (plain, 1.0e-8))
        sol = solve(IntervalNonlinearProblem{false}(f, tspan), Muller(); abstol)
        @test SciMLBase.successful_retcode(sol)
        @test abs(sol.resid) ≤ tol
    end
end
