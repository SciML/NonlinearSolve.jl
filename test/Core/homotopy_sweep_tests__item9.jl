using NonlinearSolve

using SciMLBase

H(u, p, λ) = [(1 - λ) * u[1] + λ * (u[1]^2 + 1.0)]
prob = HomotopyProblem(H, [0.0]; λspan = (0.0, 1.0))
sol = solve(prob, HomotopySweep(; adaptive = false, nsteps = 10))
@test !SciMLBase.successful_retcode(sol)
