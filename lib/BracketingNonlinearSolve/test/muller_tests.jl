@testitem "Muller" begin
    @testset "Quadratic function" begin
        f(u, p) = u^2 - p

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
        f(u, p) = sin(u)

        tspan = (1.0, 3.0)
        prob = IntervalNonlinearProblem{false}(f, tspan)
        sol = solve(prob, Muller())

        @test sol.u ≈ π

        tspan = (2.0, 6.0)
        prob = IntervalNonlinearProblem{false}(f, tspan)
        sol = solve(prob, Muller())

        @test sol.u ≈ 2*π
    end

    @testset "Exponential-sine function" begin
        f(u, p) = exp(-u)*sin(u)

        tspan = (-2.0, -4.0)
        prob = IntervalNonlinearProblem{false}(f, tspan)
        sol = solve(prob, Muller())

        @test sol.u ≈ -π

        tspan = (-3.0, 1.0)
        prob = IntervalNonlinearProblem{false}(f, tspan)
        sol = solve(prob, Muller())

        @test sol.u ≈ 0 atol = 1e-15

        tspan = (-1.0, 1.0)
        prob = IntervalNonlinearProblem{false}(f, tspan)
        sol = solve(prob, Muller())

        @test sol.u ≈ π
    end
end
