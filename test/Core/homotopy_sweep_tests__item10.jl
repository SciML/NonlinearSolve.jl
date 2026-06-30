using NonlinearSolve

using SciMLBase

# Stall guard: bisection must stop (not hang) once dλ underflows eps(λ) on a
# large-magnitude span. The λ = λspan[1] anchor is solvable from u0 (residual is
# exactly 0 there), so the sweep gets past the anchor and into continuation; every
# λ > λspan[1] is then rootless, so bisection drives dλ below eps(1e9) ≈ 1.2e-7 —
# long before the absolute sqrt(eps) floor — and the stall guard fires.
H(u, p, λ) = [u[1]^2 + (λ - 1.0e9)]
prob = HomotopyProblem(H, [0.0]; λspan = (1.0e9, 2.0e9))
sol = solve(prob, HomotopySweep(; inner = NewtonRaphson()); maxiters = 5)
@test sol.retcode == SciMLBase.ReturnCode.Stalled
@test !SciMLBase.successful_retcode(sol)
@test sol.resid === nothing
