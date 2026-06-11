using NonlinearSolve

using SciMLBase

# same family swept 1 → 0: target is the λ=0 root u = c.
H(u, p, λ) = [(1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])]
c = 4.0
prob = HomotopyProblem(H, [2.0], [c]; λspan = (1.0, 0.0))
sol = solve(prob, HomotopySweep())
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ c atol = 1.0e-6
