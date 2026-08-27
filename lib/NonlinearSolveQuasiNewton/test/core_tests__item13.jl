using NonlinearSolveQuasiNewton, NonlinearSolveBase, SciMLBase, LinearAlgebra, Test
using NonlinearSolveBase: AbsNormSafeBestTerminationMode, default_gradient_tolerance

# Broyden and Klement carry a secant approximation, not a Jacobian. The approximation is
# built only from the directions the solve has explored, so `JᵀF` computed from it can be
# small where the true gradient is not; the gradient criterion must therefore not reach
# them at all, and setting `gtol` must be inert rather than an error.
#
# Freudenstein-Roth from a start that funnels into its local minimum, posed as least
# squares: the residual is nonzero at the optimum, which is exactly where the criterion
# would fire if these solvers consulted it.
@testset "quasi-Newton solves ignore gtol: $name" for (name, alg) in (
        ("Broyden", Broyden()), ("Klement", Klement()),
    )
    fr = (u, p) -> [
        -13.0 + u[1] + ((5.0 - u[2]) * u[2] - 2.0) * u[2],
        -29.0 + u[1] + ((u[2] + 1.0) * u[2] - 14.0) * u[2],
    ]
    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction{false}(fr; resid_prototype = zeros(2)), [15.0, -2.0], nothing
    )

    on = AbsNormSafeBestTerminationMode(
        Base.Fix2(norm, 2); max_stalled_steps = 32,
        gtol = default_gradient_tolerance(Float64)
    )
    off = AbsNormSafeBestTerminationMode(Base.Fix2(norm, 2); max_stalled_steps = 32)

    a = solve(prob, alg; maxiters = 1000, termination_condition = on)
    b = solve(prob, alg; maxiters = 1000, termination_condition = off)
    @test a.retcode == b.retcode
    @test a.u ≈ b.u
    @test a.stats.nf == b.stats.nf
end
