using NonlinearSolve

using SciMLBase

# λ is not an entry of p anymore, so p needs no particular structure.
# This construction was impossible under the homotopy_parameter design.
H(u, p, λ) = [(1 - λ) * (u[1] - p.c) + λ * (u[1]^2 - p.c)]
prob = HomotopyProblem(H, [4.0], (c = 4.0,))
sol = solve(prob, HomotopySweep())
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 2.0 atol = 1.0e-6
