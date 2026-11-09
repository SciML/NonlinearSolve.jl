using BracketingNonlinearSolve
include("setup_rootfindingtestsnippet.jl")

using ADTypes: AutoEnzyme, AutoForwardDiff
using ForwardDiff: ForwardDiff, Dual, Partials, Tag, partials

const AD_ALG = NewtonBisection(; autodiff = AutoForwardDiff())

# a function barrier: on Julia 1.10 `@allocated` inside a testset counts the call's own boxing
solve_allocations(prob, alg) = @allocated solve(prob, alg; abstol = 0.0)

@testset "NewtonBisection with ForwardDiff" begin
    @testset "Roots and their derivatives" begin
        g(p) = solve(IntervalNonlinearProblem(quadratic_f, typeof(p).((1.0, 20.0)), p), AD_ALG; abstol = 1.0e-9).left
        @testset for p in 1.1:0.3:100.0
            @test g(p) ≈ sqrt(p) atol = 1.0e-3 rtol = 1.0e-3
            @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p)) atol = 1.0e-3 rtol = 1.0e-3
        end
    end

    @testset "Same steps as with the exact derivative as jac" begin
        for (f, df, tspan) in (
                ((u, p) -> u^2 - 2, (u, p) -> 2u, (1.0, 20.0)),
                ((u, p) -> exp(u) - 10, (u, p) -> exp(u), (0.0, 5.0)),
                ((u, p) -> (u - 1)^3, (u, p) -> 3 * (u - 1)^2, (0.0, 3.0)),
            )
            sol_ad = solve(IntervalNonlinearProblem(f, tspan), AD_ALG; abstol = 0.0)
            sol_jac = solve(IntervalNonlinearProblem(IntervalNonlinearFunction(f; jac = df), tspan), NewtonBisection(); abstol = 0.0)
            @test sol_ad.retcode == sol_jac.retcode
            @test (sol_ad.left, sol_ad.right) == (sol_jac.left, sol_jac.right)
        end
    end

    @testset "$T" for T in (Float32, Float64, BigFloat)
        sol = solve(IntervalNonlinearProblem(quadratic_f, T.((1, 20)), T(2)), AD_ALG; abstol = zero(T))
        @test sol.u isa T
        @test sol.retcode == ReturnCode.FloatingPointLimit
        @test nextfloat(sol.left) == sol.right
        @test sol.left ≤ sqrt(T(2)) ≤ sol.right
    end

    @testset "Float32 bracket, f promoting to Float64" begin
        sol = solve(IntervalNonlinearProblem((u, p) -> u^2 - 2.0, (1.0f0, 20.0f0)), AD_ALG; abstol = 0.0f0)
        @test SciMLBase.successful_retcode(sol)
        @test sol.left ≤ sqrt(2.0) ≤ sol.right
    end

    @testset "Allocations and inference" begin
        prob = IntervalNonlinearProblem(quadratic_f, (1.0, 20.0), 2.0)
        solve(prob, AD_ALG; abstol = 0.0)
        @test solve_allocations(prob, AD_ALG) == 0
        @test (@inferred solve(prob, AD_ALG; abstol = 0.0)) isa SciMLBase.AbstractNonlinearSolution
    end
end

@testset "No jac and no available AD backend" for alg in (NewtonBisection(), NewtonBisection(; autodiff = AutoEnzyme()))
    @test_throws ArgumentError solve(IntervalNonlinearProblem(quadratic_f, (1.0, 20.0), 2.0), alg)
end

struct NewtonBisectionADTestTag end
@testset "Duals in tspan and in f, p = $p_val" for p_val in (0.5, 2.0, 5.0)
    TTag = typeof(Tag(NewtonBisectionADTestTag(), Float64))
    p = Dual{TTag}(p_val, Partials((1.0,)))
    zero_partials = Partials((0.0,))
    tspan = (Dual{TTag}(0.0, zero_partials), Dual{TTag}(sqrt(2 / p_val) + 0.5, zero_partials))
    prob = IntervalNonlinearProblem{false}((t, _) -> 1 - p / 2 * t^2, tspan)
    sol = solve(prob, AD_ALG; abstol = 0.0)
    @test ForwardDiff.value(sol.u) ≈ sqrt(2 / p_val) atol = 1.0e-10
    @test partials(sol.u, 1) ≈ -sqrt(2) / (2 * p_val^(3 / 2)) rtol = 1.0e-3
end
