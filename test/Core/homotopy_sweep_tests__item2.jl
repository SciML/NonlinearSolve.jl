using NonlinearSolve

using SciMLBase

# f(u,p,λ) = (1-λ)*(u-c) + λ*(u^2-c); λ=0 root u=c ; λ=1 root u=√c.
H(u, p, λ) = [(1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])]
c = 4.0
prob = HomotopyProblem(H, [c], [c]; λspan = (0.0, 1.0))

sol = solve(prob, HomotopySweep())
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ sqrt(c) atol = 1.0e-6
