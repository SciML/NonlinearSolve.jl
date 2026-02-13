@testitem "Bounds: NonlinearLeastSquaresProblem" tags = [:core] begin
    using SciMLBase

    # Test out-of-place version
    f(u, p) = u .- p
    u0 = [5.0, 5.0]
    p = [1.0, 2.0]
    nf = NonlinearFunction(f; resid_prototype = zeros(2))
    alg = LevenbergMarquardt()

    @test !SciMLBase.allowsbounds(alg)

    @testset "solution within bounds" begin
        prob = NonlinearLeastSquaresProblem(nf, u0, p; lb = [0.0, 0.0], ub = [10.0, 10.0])
        sol = solve(prob, alg)

        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [1.0, 2.0] atol = 1.0e-6
        @test all(sol.u .>= 0.0)
        @test all(sol.u .<= 10.0)
    end

    @testset "solution clamped to bounds" begin
        prob = NonlinearLeastSquaresProblem(nf, u0, p; lb = [3.0, 3.0], ub = [10.0, 10.0])
        sol = solve(prob, alg)
        @test all(sol.u .>= 3.0)
        @test all(sol.u .<= 10.0)
    end

    # Test in-place version
    f!(resid, u, p) = resid .= u .- p
    u0 = [5.0, 5.0]
    p = [1.0, 2.0]
    nf = NonlinearFunction(f!)
    lb = [0.0, 0.0]
    ub = [10.0, 10.0]

    prob = NonlinearLeastSquaresProblem(nf, u0, p; lb, ub)
    sol = solve(prob, alg)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [1.0, 2.0] atol = 1.0e-6

    # Test that the original problem is preserved
    @test sol.prob.f.f === f!
    @test sol.prob.lb == lb
    @test sol.prob.ub == ub
end

@testitem "Bounds: one-sided" tags = [:core] begin
    using SciMLBase

    f(u, p) = u .- p
    u0 = [5.0, 5.0]
    p = [1.0, 2.0]
    nf = NonlinearFunction(f; resid_prototype = zeros(2))
    alg = LevenbergMarquardt()
    @test !SciMLBase.allowsbounds(alg)

    @testset "lower bound only" begin
        prob = NonlinearLeastSquaresProblem(nf, u0, p; lb = [-Inf, 0.0])
        sol = solve(prob, alg)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [1.0, 2.0] atol = 1.0e-6
    end

    @testset "upper bound only" begin
        prob = NonlinearLeastSquaresProblem(nf, u0, p; ub = [Inf, 10.0])
        sol = solve(prob, alg)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [1.0, 2.0] atol = 1.0e-6
    end

    @testset "lower bound excludes solution" begin
        prob = NonlinearLeastSquaresProblem(nf, u0, p; lb = [3.0, 3.0])
        sol = solve(prob, alg)
        @test all(sol.u .>= 3.0)
    end
end

@testitem "Bounds: nonlinear model" tags = [:core] begin
    using SciMLBase

    # A more realistic test: fit y = a*exp(b*x) with bounds on parameters
    true_a, true_b = 2.0, -0.5
    x = collect(range(0.0, 3.0; length = 20))
    y = true_a .* exp.(true_b .* x)

    model(u, p) = u[1] .* exp.(u[2] .* p) .- y
    nf = NonlinearFunction(model; resid_prototype = zeros(20))
    u0 = [1.0, -1.0]
    alg = LevenbergMarquardt()
    @test !SciMLBase.allowsbounds(alg)

    @testset "unconstrained finds true params" begin
        prob = NonlinearLeastSquaresProblem(nf, u0, x)
        sol = solve(prob, alg)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [true_a, true_b] atol = 1.0e-6
    end

    @testset "bounded finds true params when in range" begin
        prob = NonlinearLeastSquaresProblem(
            nf, u0, x; lb = [0.0, -2.0], ub = [5.0, 0.0]
        )
        sol = solve(prob, alg)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [true_a, true_b] atol = 1.0e-6
    end

    @testset "bounds constrain the solution" begin
        prob = NonlinearLeastSquaresProblem(
            nf, u0, x; lb = [3.0, -2.0], ub = [5.0, -0.1]
        )
        sol = solve(prob, alg)
        @test sol.u[1] >= 3.0
        @test sol.u[2] >= -2.0
        @test sol.u[1] <= 5.0
        @test sol.u[2] <= -0.1
    end
end
