using NonlinearSolve

using SciMLBase

# The solution path u*(λ) = 1 + λ is linear, so the secant prediction through the last
# two accepted points is exact: the corrector starts on the root and only has to verify
# convergence. The constant warm start lags by the full step, so it must spend strictly
# more residual evaluations across the (identical) sweep.
nf = Ref(0)
H(u, p, λ) = (nf[] += 1; [u[1]^3 - (1 + λ)^3])
prob = HomotopyProblem(H, [1.0]; λspan = (0.0, 1.0))

nf[] = 0
sol = solve(prob, HomotopySweep(; predictor = :secant))
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 2.0 atol = 1.0e-6
nf_secant = nf[]

nf[] = 0
sol = solve(prob, HomotopySweep(; predictor = :constant))
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 2.0 atol = 1.0e-6
nf_constant = nf[]

@test nf_secant < nf_constant
