using BracketingNonlinearSolve
include("setup_rootfindingtestsnippet.jl")

using ADTypes: AutoEnzyme, AutoForwardDiff
using ForwardDiff: ForwardDiff, Dual, Partials, Tag, partials

# With a `jac` the derivative is the user's; without, it comes from ForwardDiff.
function modab_newton_problem(f, df, tspan, p = nothing; mode)
    fun = mode === :jac ? IntervalNonlinearFunction(f; jac = df) : IntervalNonlinearFunction(f)
    return IntervalNonlinearProblem(fun, tspan, p)
end
modab_newton(mode) = mode === :jac ? ModABNewton() : ModABNewton(; autodiff = AutoForwardDiff())

# a function barrier: on Julia 1.10 `@allocated` inside a testset counts the call's own boxing
solve_allocations(prob, alg) = @allocated solve(prob, alg; abstol = 0.0)

@testset "ModABNewton, $mode" for mode in (:jac, :ad)
    alg = modab_newton(mode)

    @testset "Roots and their derivatives" begin
        function g(p)
            prob = modab_newton_problem(quadratic_f, (u, p) -> 2u, typeof(p).((1.0, 20.0)), p; mode)
            return solve(prob, alg; abstol = 1.0e-9).left
        end
        @testset for p in 1.1:0.3:100.0
            @test g(p) ≈ sqrt(p) atol = 1.0e-3 rtol = 1.0e-3
            @test ForwardDiff.derivative(g, p) ≈ 1 / (2 * sqrt(p)) atol = 1.0e-3 rtol = 1.0e-3
        end
    end

    @testset "Tolerances" begin
        prob = modab_newton_problem(quadratic_f, (u, p) -> 2u, (1.0, 20.0), 2.0; mode)
        @testset for abstol in (0.1, 1.0e-3, 1.0e-6, 1.0e-9)
            sol = solve(prob, alg; abstol)
            @test sol.retcode == ReturnCode.Success
            @test (sol.right - sol.left) / 2 < abstol
            @test sol.left ≤ sqrt(2) ≤ sol.right
        end
    end

    @testset "$T" for T in (Float32, Float64, BigFloat)
        prob = modab_newton_problem(quadratic_f, (u, p) -> 2u, T.((1, 20)), T(2); mode)
        sol = solve(prob, alg; abstol = zero(T))
        @test sol.u isa T
        @test sol.retcode == ReturnCode.FloatingPointLimit
        @test nextfloat(sol.left) == sol.right
        @test sol.left ≤ sqrt(T(2)) ≤ sol.right
    end

    @testset "Hard cases" begin
        @testset for (f, df, tspan) in (
                ((u, p) -> (u - 1)^3, (u, p) -> 3 * (u - 1)^2, (0.0, 3.0)),   # triple root
                ((u, p) -> u^20 - 1, (u, p) -> 20u^19, (0.0, 5.0)),   # flat on the left
                ((u, p) -> atan(u), (u, p) -> 1 / (1 + u^2), (-10.0, 20.0)),   # Newton leaves the bracket
                ((u, p) -> u^3 - 2, (u, p) -> 3u^2, (0.0, 2.0)),   # zero derivative at an end
                ((u, p) -> max(u, 3u - 1) - 0.8, (u, p) -> u < 0.5 ? 1.0 : 3.0, (0.0, 1.0)),   # kink
                ((u, p) -> u < 1 / 3 ? -1.0 : 1.0, (u, p) -> 0.0, (0.0, 1.0)),   # jump, f' = 0
            )
            sol = solve(modab_newton_problem(f, df, tspan; mode), alg; abstol = 0.0)
            @test SciMLBase.successful_retcode(sol)
            @test iszero(f(sol.u, nothing)) ||
                (sign(f(sol.left, nothing)) != sign(f(sol.right, nothing)) && nextfloat(sol.left) == sol.right)
        end
    end

    @testset "Allocations and inference" begin
        prob = modab_newton_problem(quadratic_f, (u, p) -> 2u, (1.0, 20.0), 2.0; mode)
        solve(prob, alg; abstol = 0.0)
        @test solve_allocations(prob, alg) == 0
        @test (@inferred solve(prob, alg; abstol = 0.0)) isa SciMLBase.AbstractNonlinearSolution
    end
end

@testset "ModABNewton evaluates the derivative less often than NewtonBisection" for (f, df, tspan) in (
        (u -> u^2 - 2, u -> 2u, (1.0, 20.0)),
        (u -> exp(u) - 10, u -> exp(u), (0.0, 5.0)),
        (u -> atan(u) - 1, u -> 1 / (1 + u^2), (-10.0, 20.0)),
    )
    function jac_calls(alg)
        calls = Ref(0)
        fun = IntervalNonlinearFunction((u, p) -> f(u); jac = (u, p) -> (calls[] += 1; df(u)))
        solve(IntervalNonlinearProblem(fun, tspan), alg; abstol = 0.0)
        return calls[]
    end
    @test jac_calls(ModABNewton()) < jac_calls(NewtonBisection())
end

@testset "No jac and no available AD backend" for alg in (ModABNewton(), ModABNewton(; autodiff = AutoEnzyme()))
    @test_throws ArgumentError solve(IntervalNonlinearProblem(quadratic_f, (1.0, 20.0), 2.0), alg)
end

struct ModABNewtonTestTag end
@testset "Duals in tspan and in f, $mode, p = $p_val" for mode in (:jac, :ad), p_val in (0.5, 2.0, 5.0)
    TTag = typeof(Tag(ModABNewtonTestTag(), Float64))
    p = Dual{TTag}(p_val, Partials((1.0,)))
    zero_partials = Partials((0.0,))
    tspan = (Dual{TTag}(0.0, zero_partials), Dual{TTag}(sqrt(2 / p_val) + 0.5, zero_partials))
    f = (t, _) -> 1 - p / 2 * t^2
    fun = mode === :jac ? IntervalNonlinearFunction{false}(f; jac = (t, _) -> -p * t) :
        IntervalNonlinearFunction{false}(f)
    sol = solve(IntervalNonlinearProblem{false}(fun, tspan), modab_newton(mode); abstol = 0.0)
    @test ForwardDiff.value(sol.u) ≈ sqrt(2 / p_val) atol = 1.0e-10
    @test partials(sol.u, 1) ≈ -sqrt(2) / (2 * p_val^(3 / 2)) rtol = 1.0e-3
end
