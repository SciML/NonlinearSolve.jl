using NonlinearSolve

using SciMLBase

# Monotone (fold-free) homotopy: arclength continuation must reproduce the natural sweep's
# answer on the easy case. u* = 2 at λ = 1.
H(u, p, λ) = [(1 - λ) * (u[1] - p.c) + λ * (u[1]^2 - p.c)]
prob = HomotopyProblem(H, [4.0], (c = 4.0,))

sol = solve(prob, ArcLengthContinuation())
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 2.0 atol = 1.0e-6

# the result is on the target system H(u, 1) = u^2 - c
@test abs(sol.u[1]^2 - 4.0) < 1.0e-8

# matches HomotopySweep on the same problem
ref = solve(prob, HomotopySweep())
@test sol.u[1] ≈ ref.u[1] atol = 1.0e-6

# decreasing λspan is handled (target is the λspan[2] end, here λ = 0 ⇒ linear system u = c)
prob_dec = HomotopyProblem(H, [1.0], (c = 4.0,); λspan = (1.0, 0.0))
sol_dec = solve(prob_dec, ArcLengthContinuation())
@test SciMLBase.successful_retcode(sol_dec)
@test sol_dec.u[1] ≈ 4.0 atol = 1.0e-6
