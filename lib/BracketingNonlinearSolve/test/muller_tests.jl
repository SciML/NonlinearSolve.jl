@testitem "Muller" begin
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

        @test sol.u≈0 atol=1e-15

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
end
