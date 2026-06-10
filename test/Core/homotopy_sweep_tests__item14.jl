using NonlinearSolve

using SciMLBase

c = 4.0
H(u, p, λ) = [(1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])]
prob = HomotopyProblem(H, [c], [c]; λspan = (0.0, 1.0))
sol = solve(prob, nothing)        # no algorithm → should default to HomotopySweep
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ sqrt(c) atol = 1.0e-6

sol2 = solve(prob)                # zero-arg form must route the same way
@test SciMLBase.successful_retcode(sol2)
@test sol2.u[1] ≈ sqrt(c) atol = 1.0e-6
