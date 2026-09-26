using NonlinearSolve

using SciMLBase, StaticArrays

# Test with in-place function
function f_iip(du, u, p, t)
    du[1] = 2 - 2u[1]
    return du[2] = u[1] - 4u[2]
end

u0 = zeros(2)
prob_iip = SteadyStateProblem(f_iip, u0)

@testset "In-place SteadyStateProblem" begin
    # Test with default algorithm (nothing)
    sol = solve(prob_iip)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test maximum(abs, sol.resid) < 1.0e-6

    # Test with explicit nothing
    sol = solve(prob_iip, nothing)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test maximum(abs, sol.resid) < 1.0e-6

    # Test init interface
    cache = init(prob_iip)
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test maximum(abs, sol.resid) < 1.0e-6

    # Test init with nothing
    cache = init(prob_iip, nothing)
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test maximum(abs, sol.resid) < 1.0e-6
end

# Test with out-of-place function
f_oop(u, p, t) = [2 - 2u[1], u[1] - 4u[2]]
u0 = zeros(2)
prob_oop = SteadyStateProblem(f_oop, u0)

@testset "Out-of-place SteadyStateProblem" begin
    # Test with default algorithm (nothing)
    sol = solve(prob_oop)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test maximum(abs, sol.resid) < 1.0e-6

    # Test with explicit nothing
    sol = solve(prob_oop, nothing)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test maximum(abs, sol.resid) < 1.0e-6

    # Test init interface
    cache = init(prob_oop)
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test maximum(abs, sol.resid) < 1.0e-6

    # Test init with nothing
    cache = init(prob_oop, nothing)
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test maximum(abs, sol.resid) < 1.0e-6
end

# Test that SteadyStateProblem conversion works
@testset "Problem conversion" begin
    # Create equivalent NonlinearProblem
    function f_nl(u, p)
        [2 - 2u[1], u[1] - 4u[2]]
    end

    prob_nl = NonlinearProblem(f_nl, u0)

    # Convert SteadyStateProblem to NonlinearProblem
    prob_converted = NonlinearProblem(prob_oop)

    # Both should solve to the same solution
    sol_nl = solve(prob_nl)
    sol_converted = solve(prob_converted)

    @test sol_nl.u ≈ sol_converted.u atol = 1.0e-10
end

# Test with StaticArrays
@testset "StaticArrays support" begin
    f_static(u, p, t) = @SVector [2 - 2u[1], u[1] - 4u[2]]
    u0_static = @SVector [0.0, 0.0]
    prob_static = SteadyStateProblem(f_static, u0_static)

    sol = solve(prob_static)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test maximum(abs, sol.resid) < 1.0e-6
end

# Test that solve works with various problem types
@testset "Mixed problem types" begin
    # Regular arrays
    prob1 = SteadyStateProblem(f_oop, [0.5, 0.5])
    sol1 = solve(prob1)
    @test SciMLBase.successful_retcode(sol1.retcode)

    # With parameters
    f_param(u, p, t) = [p[1] - 2u[1], u[1] - 4u[2]]
    prob2 = SteadyStateProblem(f_param, [0.5, 0.5], [2.0])
    sol2 = solve(prob2)
    @test SciMLBase.successful_retcode(sol2.retcode)
end

# SciML/ModelingToolkit.jl#5194; `lowered_problem` needs SciMLBase >= 3.55
if :lowered_problem in fieldnames(SteadyStateProblem)
    @testset "SteadyStateProblem with SCC lowering" begin
        lowered = SciMLBase.SCCNonlinearProblem(
            (NonlinearProblem((u, p) -> 2 .- 2u, [0.0]),), (Returns(nothing),)
        )
        prob = SteadyStateProblem(f_iip, zeros(2); lowered_problem = lowered)
        @test SciMLBase.NonlinearProblem(prob) isa SciMLBase.SCCNonlinearProblem

        for cache in (init(prob), init(prob, nothing))
            sol = solve!(cache)
            @test SciMLBase.successful_retcode(sol.retcode)
            @test sol.u ≈ [1.0, 0.25]
        end
    end
end

@testset "ImmutableNonlinearProblem default algorithm" begin
    prob = SciMLBase.ImmutableNonlinearProblem{false}(
        NonlinearFunction{false}((u, p) -> u .* u .- p), [1.0, 1.0], 2.0
    )
    for sol in (
            solve(prob), solve(prob, nothing), solve!(init(prob)), solve!(init(prob, nothing)),
        )
        @test SciMLBase.successful_retcode(sol.retcode)
        @test sol.u ≈ fill(sqrt(2.0), 2)
    end
end
