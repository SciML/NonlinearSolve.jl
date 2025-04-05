@testitem "SimpleMuller" begin
    @testset "Quadratic function" begin
        f(u, p) = u^2 - p

        u0 = (10.0, 20.0, 30.0)
        p = 612.0
        prob = NonlinearProblem{false}(f, u0, p)
        sol = solve(prob, SimpleMuller())

        @test sol.u ≈ √612

        u0 = (-10.0, -20.0, -30.0)
        prob = NonlinearProblem{false}(f, u0, p)
        sol = solve(prob, SimpleMuller())

        @test sol.u ≈ -√612
    end

    @testset "Sine function" begin
        f(u, p) = sin(u)

        u0 = (1.0, 2.0, 3.0)
        prob = NonlinearProblem{false}(f, u0)
        sol = solve(prob, SimpleMuller())

        @test sol.u ≈ π

        u0 = (2.0, 4.0, 6.0)
        prob = NonlinearProblem{false}(f, u0)
        sol = solve(prob, SimpleMuller())

        @test sol.u ≈ 2*π
    end

    @testset "Exponential-sine function" begin
        f(u, p) = exp(-u)*sin(u)

        u0 = (-2.0, -3.0, -4.0)
        prob = NonlinearProblem{false}(f, u0)
        sol = solve(prob, SimpleMuller())

        @test sol.u ≈ -π

        u0 = (-1.0, 0.0, 1/2)
        prob = NonlinearProblem{false}(f, u0)
        sol = solve(prob, SimpleMuller())

        @test sol.u ≈ 0

        u0 = (-1.0, 0.0, 1.0)
        prob = NonlinearProblem{false}(f, u0)
        sol = solve(prob, SimpleMuller())

        @test sol.u ≈ π
    end
end
