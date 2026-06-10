using NonlinearSolve

using SciMLBase

H(u, p, λ) = [(1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])]
prob = HomotopyProblem(H, [4.0f0], [4.0f0]; λspan = (0.0f0, 1.0f0))
sol = solve(prob, HomotopySweep())
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 2.0f0 atol = 1.0e-3
