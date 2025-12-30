@testsnippet RootfindingTestSnippet begin
    linear_f(u, p) = u .- p
    quadratic_f(u, p) = u .* u .- p
    quadratic_f!(du, u, p) = (du .= u .* u .- p)
    quadratic_f2(u, p) = @. p[1] * u * u - p[2]
end

@testitem "Interval Nonlinear Problems" setup=[RootfindingTestSnippet] tags=[:core] begin
    using ForwardDiff

    @testset for alg in (
        Alefeld(), Bisection(), Brent(), Falsi(), ITP(), Muller(), Ridder(), nothing)
        tspan = (1.0, 20.0)

        function g(p)
            probN = IntervalNonlinearProblem{false}(quadratic_f, typeof(p).(tspan), p)
            return solve(probN, alg; abstol = 1e-9).left
        end

        @testset for p in 1.1:0.1:100.0
            @test g(p)≈sqrt(p) atol=1e-3 rtol=1e-3
            @test ForwardDiff.derivative(g, p)≈1/(2*sqrt(p)) atol=1e-3 rtol=1e-3
        end

        t = (p) -> [sqrt(p[2] / p[1])]
        p = [0.9, 50.0]

        function g2(p)
            probN = IntervalNonlinearProblem{false}(quadratic_f2, tspan, p)
            sol = solve(probN, alg; abstol = 1e-9)
            return [sol.u]
        end

        @test g2(p)≈[sqrt(p[2]/p[1])] atol=1e-3 rtol=1e-3
        @test ForwardDiff.jacobian(g2, p)≈ForwardDiff.jacobian(t, p) atol=1e-3 rtol=1e-3

        probB = IntervalNonlinearProblem{false}(quadratic_f, (1.0, 2.0), 2.0)
        sol = solve(probB, alg; abstol = 1e-9)
        @test sol.left≈sqrt(2.0) atol=1e-3 rtol=1e-3

        if !(alg isa Bisection || alg isa Falsi)
            probB = IntervalNonlinearProblem{false}(quadratic_f, (sqrt(2.0), 10.0), 2.0)
            sol = solve(probB, alg; abstol = 1e-9)
            @test sol.left≈sqrt(2.0) atol=1e-3 rtol=1e-3

            probB = IntervalNonlinearProblem{false}(quadratic_f, (0.0, sqrt(2.0)), 2.0)
            sol = solve(probB, alg; abstol = 1e-9)
            @test sol.left≈sqrt(2.0) atol=1e-3 rtol=1e-3
        end
    end
end

@testitem "Tolerance Tests Interval Methods" setup=[RootfindingTestSnippet] tags=[:core] begin
    prob=IntervalNonlinearProblem(quadratic_f, (1.0, 20.0), 2.0)
    ϵ=eps(Float64) # least possible tol for all methods

    @testset for alg in (Bisection(), Falsi(), ITP(), Muller(), nothing)
        @testset for abstol in [0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6]
            sol = solve(prob, alg; abstol)
            result_tol = abs(sol.u - sqrt(2))
            @test result_tol < abstol
            # test that the solution is not calculated upto max precision
            @test result_tol > ϵ
        end
    end

    @testset for alg in (Brent(), Ridder())
        # Ridder and Brent converge rapidly so as we lower tolerance below 0.01, it
        # converges with max precision to the solution
        @testset for abstol in [0.1]
            sol = solve(prob, alg; abstol)
            result_tol = abs(sol.u - sqrt(2))
            @test result_tol < abstol
            # test that the solution is not calculated upto max precision
            @test result_tol > ϵ
        end
    end
end

@testitem "Zero-Tolerance Tests Interval Methods" setup=[RootfindingTestSnippet] tags=[:core] begin
    prob=IntervalNonlinearProblem(quadratic_f, (1.0, 20.0), 2.0)
    prob_lin=IntervalNonlinearProblem(linear_f, (-1.0, 1.0), 0.0)

    @testset for alg in (ITP(),)
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
end

@testitem "Flipped Signs and Reversed Tspan" setup=[RootfindingTestSnippet] tags=[:core] begin
    @testset for alg in (
        Alefeld(), Bisection(), Brent(), Falsi(), ITP(), Muller(), Ridder(), nothing)
        f1(u, p) = u * u - p
        f2(u, p) = p - u * u

        for p in 1:4
            sol1 = solve(IntervalNonlinearProblem(f1, (1.0, 2.0), p), alg)
            sol2 = solve(IntervalNonlinearProblem(f2, (1.0, 2.0), p), alg)
            sol3 = solve(IntervalNonlinearProblem(f1, (2.0, 1.0), p), alg)
            sol4 = solve(IntervalNonlinearProblem(f2, (2.0, 1.0), p), alg)
            @test abs.(sol1.u) ≈ sqrt.(p)
            @test abs.(sol2.u) ≈ sqrt.(p)
            @test abs.(sol3.u) ≈ sqrt.(p)
            @test abs.(sol4.u) ≈ sqrt.(p)
            # Test brackets consistency
            @test sol1.left ≤ sol1.right
            @test sol2.left ≤ sol2.right
            @test sol3.left ≥ sol3.right
            @test sol4.left ≥ sol4.right
        end
    end
end
