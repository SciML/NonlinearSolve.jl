using BracketingNonlinearSolve

using ForwardDiff: ForwardDiff, Dual, Partials, Tag, partials

# Mixed case: tspan has Duals AND f's closure captures Dual-valued state.
# The derivative w.r.t. the boundary point is zero, but dt*/dθ is nonzero
# via the implicit function theorem: dt*/dθ = -(∂f/∂θ)/(∂f/∂t).
# Uses a proper ForwardDiff Tag (not Nothing) so nested Dual ordering works.
struct TestMixedTag end
@testset "mixed: closure captures Duals" for alg in (
        Alefeld(), Bisection(), Brent(), Falsi(), ITP(), Ridder(), ModAB(), nothing,
    )
    @testset for p_val in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        TTag = typeof(Tag(TestMixedTag(), Float64))
        p = Dual{TTag}(p_val, Partials((1.0,)))
        exact_root = sqrt(2 / p_val)
        exact_deriv = -sqrt(2) / (2 * p_val^(3 / 2))

        cond(t, _ = nothing) = 1 - p / 2 * t^2

        t_left = Dual{TTag}(0.0, Partials((0.0,)))
        t_right = Dual{TTag}(exact_root + 0.5, Partials((0.0,)))

        prob = IntervalNonlinearProblem{false}(cond, (t_left, t_right))
        sol = solve(prob, alg; abstol = 0.0, reltol = 0.0)

        @test ForwardDiff.value(sol.u) ≈ exact_root atol = 1.0e-10
        @test partials(sol.u, 1) ≈ exact_deriv rtol = 1.0e-3
        @test all(isfinite, [partials(sol.left, 1), partials(sol.right, 1)])
    end
end

# Pure tspan-Dual case: f returns plain floats, derivative w.r.t. boundary is zero.
struct TestPureTag end
@testset "pure: f returns Float64" for alg in (
        Alefeld(), Bisection(), Brent(), Falsi(), ITP(), Ridder(), ModAB(), nothing,
    )
    f_plain(t, _ = nothing) = t^2 - 2.0

    TTag = typeof(Tag(TestPureTag(), Float64))
    t_left = Dual{TTag}(0.0, Partials((1.0,)))
    t_right = Dual{TTag}(3.0, Partials((1.0,)))

    prob = IntervalNonlinearProblem{false}(f_plain, (t_left, t_right))
    sol = solve(prob, alg; abstol = 0.0, reltol = 0.0)

    @test ForwardDiff.value(sol.u) ≈ sqrt(2.0) atol = 1.0e-10
    @test partials(sol.u, 1) == 0.0
    @test partials(sol.left, 1) == 0.0
    @test partials(sol.right, 1) == 0.0
end
