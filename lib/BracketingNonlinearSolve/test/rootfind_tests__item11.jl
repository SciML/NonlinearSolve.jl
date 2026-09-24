using BracketingNonlinearSolve
include("setup_rootfindingtestsnippet.jl")

using ForwardDiff: ForwardDiff, Dual, Partials, Tag, partials

quadratic_fn = IntervalNonlinearFunction(quadratic_f; jac = (u, p) -> 2u)

@testset "Roots and their derivatives" begin
    function g(p)
        prob = IntervalNonlinearProblem{false}(quadratic_fn, typeof(p).((1.0, 20.0)), p)
        return solve(prob, NewtonBisection(); abstol = 1.0e-9).left
    end

    @testset for p in 1.1:0.1:100.0
        @test g(p) ≈ sqrt(p) atol = 1.0e-3 rtol = 1.0e-3
        @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p)) atol = 1.0e-3 rtol = 1.0e-3
    end
end

@testset "Tolerances" begin
    prob = IntervalNonlinearProblem(quadratic_fn, (1.0, 20.0), 2.0)
    @testset for abstol in (0.1, 1.0e-3, 1.0e-6, 1.0e-9)
        sol = solve(prob, NewtonBisection(); abstol)
        @test sol.retcode == ReturnCode.Success
        @test (sol.right - sol.left) / 2 < abstol
        @test sol.left ≤ sqrt(2) ≤ sol.right
    end

    sol = solve(prob, NewtonBisection(); abstol = 0.0)
    @test sol.retcode == ReturnCode.FloatingPointLimit
    @test quadratic_f(sol.left, 2.0) < 0 < quadratic_f(sol.right, 2.0)
    @test nextfloat(sol.left) == sol.right

    linear_fn = IntervalNonlinearFunction(linear_f; jac = (u, p) -> one(u))
    sol = solve(IntervalNonlinearProblem(linear_fn, (-1.0, 1.0), 0.0), NewtonBisection(); abstol = 0.0)
    @test sol.retcode == ReturnCode.Success
    @test sol.u == sol.left == sol.right == 0.0
end

@testset "Flipped signs and reversed tspan" begin
    f1 = IntervalNonlinearFunction((u, p) -> u * u - p; jac = (u, p) -> 2u)
    f2 = IntervalNonlinearFunction((u, p) -> p - u * u; jac = (u, p) -> -2u)
    for p in 1:4
        sol1 = solve(IntervalNonlinearProblem(f1, (1.0, 2.0), p), NewtonBisection())
        sol2 = solve(IntervalNonlinearProblem(f2, (1.0, 2.0), p), NewtonBisection())
        sol3 = solve(IntervalNonlinearProblem(f1, (2.0, 1.0), p), NewtonBisection())
        sol4 = solve(IntervalNonlinearProblem(f2, (2.0, 1.0), p), NewtonBisection())
        @test sol1.u ≈ sqrt(p)
        @test sol2.u ≈ sqrt(p)
        @test sol3.u ≈ sqrt(p)
        @test sol4.u ≈ sqrt(p)
        @test sol1.left ≤ sol1.right
        @test sol2.left ≤ sol2.right
        @test sol3.left ≥ sol3.right
        @test sol4.left ≥ sol4.right
    end
end

@testset "Hard cases" begin
    @testset for (f, df, tspan) in (
            ((u, p) -> (u - 1)^3, (u, p) -> 3 * (u - 1)^2, (0.0, 3.0)),   # triple root
            ((u, p) -> atan(u), (u, p) -> 1 / (1 + u^2), (-10.0, 20.0)),   # Newton leaves the bracket
            ((u, p) -> u^3 - 2, (u, p) -> 3u^2, (0.0, 2.0)),   # zero derivative at an end
        )
        sol = solve(IntervalNonlinearProblem(IntervalNonlinearFunction(f; jac = df), tspan), NewtonBisection(); abstol = 0.0)
        @test SciMLBase.successful_retcode(sol)
        @test iszero(f(sol.u, nothing)) || sign(f(sol.left, nothing)) != sign(f(sol.right, nothing))
    end
end

@testset "$T" for T in (Float32, BigFloat)
    sol = solve(IntervalNonlinearProblem(quadratic_fn, T.((1, 20)), T(2)), NewtonBisection(); abstol = zero(T))
    @test sol.retcode == ReturnCode.FloatingPointLimit
    @test nextfloat(sol.left) == sol.right
    @test sol.left ≤ sqrt(T(2)) ≤ sol.right
end

@testset "Fewer evaluations than ModAB" begin
    @testset for (f, df, tspan) in (
            ((u, p) -> u^2 - 2, (u, p) -> 2u, (1.0, 20.0)),
            ((u, p) -> exp(u) - 10, (u, p) -> exp(u), (0.0, 5.0)),
            ((u, p) -> cos(u) - u, (u, p) -> -sin(u) - 1, (0.0, 1.0)),
        )
        calls = Ref(0)
        counted = (u, p) -> (calls[] += 1; f(u, p))
        solve(IntervalNonlinearProblem(counted, tspan), ModAB(); abstol = 0.0)
        modab = calls[]
        calls[] = 0
        prob = IntervalNonlinearProblem(IntervalNonlinearFunction(counted; jac = df), tspan)
        solve(prob, NewtonBisection(); abstol = 0.0)
        @test calls[] < modab
    end
end

@testset "Missing derivative" begin
    prob = IntervalNonlinearProblem(quadratic_f, (1.0, 20.0), 2.0)
    @test_throws ArgumentError solve(prob, NewtonBisection())
end

struct NewtonBisectionTestTag end
@testset "Duals in tspan and in f" for p_val in (0.5, 2.0, 5.0)
    TTag = typeof(Tag(NewtonBisectionTestTag(), Float64))
    p = Dual{TTag}(p_val, Partials((1.0,)))
    f = IntervalNonlinearFunction{false}((t, _) -> 1 - p / 2 * t^2; jac = (t, _) -> -p * t)
    zero_partials = Partials((0.0,))
    tspan = (Dual{TTag}(0.0, zero_partials), Dual{TTag}(sqrt(2 / p_val) + 0.5, zero_partials))

    sol = solve(IntervalNonlinearProblem{false}(f, tspan), NewtonBisection(); abstol = 0.0)
    @test ForwardDiff.value(sol.u) ≈ sqrt(2 / p_val) atol = 1.0e-10
    @test partials(sol.u, 1) ≈ -sqrt(2) / (2 * p_val^(3 / 2)) rtol = 1.0e-3
end
