using NonlinearSolve

using SciMLBase

H!(du, u, p, λ) = (du[1] = (1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1]); nothing)
prob = HomotopyProblem(H!, [4.0], [4.0])
sol = solve(prob, HomotopySweep())
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 2.0 atol = 1.0e-6
