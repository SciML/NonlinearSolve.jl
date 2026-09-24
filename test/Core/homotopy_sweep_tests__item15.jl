using NonlinearSolve

using SciMLBase

# Regression for the λ = λspan[1] anchor. The sweep must solve the `simplified`
# system FIRST; otherwise a poor u0 converges onto the wrong branch at the initial
# λ0 + dλ step and the continuation tracks that branch to a WRONG root with a
# success retcode.
#   actual     u^2 - 4   has roots ±2;
#   simplified u   - 4    is linear (root +4), solvable from ANY guess and in the
#                         basin of the POSITIVE root.
# Anchoring at λ = 0 lands the sweep on +2. Without the anchor, the first solve at
# λ = 0.1 from u0 = -10 converges to the negative branch and the sweep returns -2.
H(u, p, λ) = [(1 - λ) * (u[1] - 4.0) + λ * (u[1]^2 - 4.0)]
prob = HomotopyProblem(H, [-10.0]; λspan = (0.0, 1.0))
sol = solve(prob, HomotopySweep())
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 2.0 atol = 1.0e-5    # +2 via the λ=0 anchor (pre-fix: -2)
