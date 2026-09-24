using NonlinearSolve

using SciMLBase

# Zero-width λspan: the anchor IS the single target solve, so the sweep solves the
# system once and returns success (the degenerate case the anchor subsumes; the
# continuation loop is never entered).
H(u, p, λ) = [u[1]^2 - 4.0]
prob = HomotopyProblem(H, [1.5]; λspan = (0.7, 0.7))
sol = solve(prob, HomotopySweep(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 2.0 atol = 1.0e-8
