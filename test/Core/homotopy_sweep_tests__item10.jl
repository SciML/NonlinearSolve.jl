using NonlinearSolve

using SciMLBase

# rootless residual on a large-magnitude span: bisection drives dλ below
# eps(λ) ≈ 1.2e-7 long before the absolute sqrt(eps) floor stops it, so the
# stall guard must fire instead of looping forever.
H(u, p, λ) = [u[1]^2 + 1.0]
prob = HomotopyProblem(H, [0.0]; λspan = (1.0e9, 2.0e9))
sol = solve(prob, HomotopySweep(; inner = NewtonRaphson()); maxiters = 5)
@test sol.retcode == SciMLBase.ReturnCode.Stalled
@test !SciMLBase.successful_retcode(sol)
@test sol.resid === nothing
