using NonlinearSolve

using SciMLBase

# Anchor-failure contract. If the λ = λspan[1] system is itself unsolvable from u0,
# the homotopy premise is broken: the sweep fails fast with the inner failure
# retcode (NOT Stalled), a non-`nothing` resid, and `u` left at u0 — rather than
# silently stepping past the unsolved anchor.
H(u, p, λ) = [u[1]^2 + 1.0]    # rootless at every λ, including λ = λspan[1]
prob = HomotopyProblem(H, [0.5]; λspan = (0.0, 1.0))
sol = solve(prob, HomotopySweep(; inner = NewtonRaphson()); maxiters = 25)
@test !SciMLBase.successful_retcode(sol)
@test sol.retcode != SciMLBase.ReturnCode.Stalled    # anchor failure, not a stall
@test sol.resid !== nothing
@test sol.u[1] ≈ 0.5                                  # u0 unchanged
