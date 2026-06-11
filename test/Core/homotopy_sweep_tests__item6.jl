using NonlinearSolve

using SciMLBase

# No real root for λ>1/3 (fold): (1-λ)*u + λ*(u^2 + 1). Continuation cannot reach λ=1.
H(u, p, λ) = [(1 - λ) * u[1] + λ * (u[1]^2 + 1.0)]
prob = HomotopyProblem(H, [0.0]; λspan = (0.0, 1.0))
sol = solve(prob, HomotopySweep(; min_dλ = 1.0e-2))     # explicit floor keeps the test fast
@test !SciMLBase.successful_retcode(sol)              # must fail, not silently "succeed"
@test sol.retcode != SciMLBase.ReturnCode.Success
@test all(isfinite, sol.u)    # failure returns the last CONVERGED iterate, not a diverged buffer
