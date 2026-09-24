using NonlinearSolve

using SciMLBase

# actual residual atan(u-3) has root u=3 but its derivative saturates, so a cold
# Newton from u0=12 overshoots/diverges. simplified residual u has root u=0.
H(u, p, λ) = [(1 - λ) * u[1] + λ * atan(u[1] - 3.0)]
prob = HomotopyProblem(H, [12.0]; λspan = (0.0, 1.0))

# nsteps with adaptive=true sets the INITIAL step (span/nsteps); bisection stays on
sol = solve(prob, HomotopySweep(; nsteps = 20))
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 3.0 atol = 1.0e-5

# contrast: cold Newton on the actual system from the same guess does NOT land on 3.
cold = NonlinearProblem((u, p) -> [atan(u[1] - 3.0)], [12.0])
csol = solve(cold, NewtonRaphson())
@test !(SciMLBase.successful_retcode(csol) && isapprox(csol.u[1], 3.0; atol = 1.0e-3))
