using NonlinearSolve

using SciMLBase
using Enzyme
using NonlinearSolveBase: transform_bounded_problem, BoundedWrapper
using PreallocationTools: FixedSizeDiffCache
using ADTypes: AutoForwardDiff, AutoEnzyme

# Test that the right cache type is chosen for the different autodiff backends
let prob = NonlinearLeastSquaresProblem(Returns(42), [1.0, 1.0], 1:10)
    # Default to ForwardDiff
    unbounded_prob = transform_bounded_problem(prob, nothing)
    wrapper = unbounded_prob.f.f
    @test wrapper isa BoundedWrapper
    @test wrapper.u_cache isa FixedSizeDiffCache

    unbounded_prob = transform_bounded_problem(prob, LevenbergMarquardt())
    @test unbounded_prob.f.f.u_cache isa FixedSizeDiffCache

    unbounded_prob = transform_bounded_problem(prob, LevenbergMarquardt(; autodiff = AutoForwardDiff()))
    @test unbounded_prob.f.f.u_cache isa FixedSizeDiffCache

    # Test Enzyme
    unbounded_prob = transform_bounded_problem(prob, LevenbergMarquardt(; autodiff = AutoEnzyme()))
    @test unbounded_prob.f.f.u_cache isa typeof(prob.u0)
end

for autodiff in (AutoEnzyme(; function_annotation = Enzyme.Duplicated), AutoForwardDiff())
    # A more realistic test: fit y = a*exp(b*x) with bounds on parameters
    true_a, true_b = 2.0, -0.5
    x = collect(range(0.0, 3.0; length = 20))
    y = true_a .* exp.(true_b .* x)

    model(u, p) = u[1] .* exp.(u[2] .* p) .- y
    nf = NonlinearFunction(model; resid_prototype = zeros(20))
    u0 = [1.0, -1.0]
    alg = LevenbergMarquardt(; autodiff)
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
