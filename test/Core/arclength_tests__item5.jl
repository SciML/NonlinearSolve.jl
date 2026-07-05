using NonlinearSolve

using SciMLBase

# The solution curve is the circle u^2 + (λ - 0.5)^2 = 0.25, which only exists for
# λ ∈ [0, 1]. A target at λspan[2] = 2 is off the curve and unreachable: arclength
# continuation must report a failure retcode (and terminate — not hang — thanks to the
# maxsteps guard) rather than claim success.
Hc(u, p, λ) = [u[1]^2 + (λ - 0.5)^2 - 0.25]
prob = HomotopyProblem(Hc, [0.0]; λspan = (0.0, 2.0))
sol = solve(prob, ArcLengthContinuation(; maxsteps = 200))
@test !SciMLBase.successful_retcode(sol)
@test all(isfinite, sol.u)            # last converged curve point, not a diverged buffer

# A start the corrector cannot bring onto the curve also fails cleanly: at λ0 = 0 the
# circle is tangent (u = 0); start far away with a target system that has no real root at
# λ = 1 — here the embedded system stays solvable, so instead test that a tiny maxsteps
# forces termination on an otherwise-solvable fold problem.
H(u, p, λ) = [u[1]^3 - 3 * u[1] - (-3 + 6λ)]
probf = HomotopyProblem(H, [-2.1038]; λspan = (0.0, 1.0))
sol_short = solve(probf, ArcLengthContinuation(; maxsteps = 2))
@test !SciMLBase.successful_retcode(sol_short)   # 2 attempts cannot round the fold
@test all(isfinite, sol_short.u)
