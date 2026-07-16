using NonlinearSolve

using SciMLBase

# A `HomotopyProblem` handed a *standard* (non-continuation) nonlinear algorithm routes to
# the `HomotopyPolyAlgorithm` default. This lets init paths that select a default nonlinear
# solver for the problem (SciMLBase OverrideInit, SteadyStateDiffEq, ...) — which see a
# `HomotopyProblem` as the `AbstractNonlinearProblem` it subtypes — solve it by continuation.

# out-of-basin guess: `atan` saturates, so a plain Newton from u0 = 12 diverges, but the
# continuation from the simplified `u` (root 0) walks to the true root 3.
Hatan(u, p, λ) = [(1 - λ) * u[1] + λ * atan(u[1] - 3)]
prob = HomotopyProblem(Hatan, [12.0]; λspan = (0.0, 1.0))

# --- standard nonlinear algorithms are rerouted to continuation and succeed ---
for alg in (NewtonRaphson(), TrustRegion(), FastShortcutNonlinearPolyalg())
    sol = solve(prob, alg; abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 3.0 atol = 1.0e-6
end

# the fallback genuinely routes to the polyalgorithm: same answer as solving with it directly
sol_fb = solve(prob, NewtonRaphson())
sol_poly = solve(prob, HomotopyPolyAlgorithm())
@test sol_fb.u[1] ≈ sol_poly.u[1] atol = 1.0e-8

# --- the continuation algorithms keep their own, more specific dispatch (no recursion) ---
for alg in (HomotopySweep(), ArcLengthContinuation(), HomotopyPolyAlgorithm())
    sol = solve(prob, alg; abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 3.0 atol = 1.0e-6
end
