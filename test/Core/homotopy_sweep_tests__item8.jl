using NonlinearSolve

using SciMLBase

# maxiters = 1 stored on the problem must reach the inner solves and wreck them;
# before prob.kwargs forwarding this succeeded by silently ignoring it.
H(u, p, λ) = [(1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])]
prob = HomotopyProblem(H, [4.0], [4.0]; maxiters = 1)
sol = solve(prob, HomotopySweep(; inner = NewtonRaphson(), min_dλ = 1.0e-2))
@test !SciMLBase.successful_retcode(sol)
