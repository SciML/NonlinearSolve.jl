using NonlinearSolve

using SciMLBase

c = 4.0
H(u, p, λ) = [(1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])]
mkprob() = HomotopyProblem(H, [c], [c]; λspan = (0.0, 1.0))

# default inner (nothing → polyalgorithm)
s_default = solve(mkprob(), HomotopySweep())
@test SciMLBase.successful_retcode(s_default)
@test s_default.u[1] ≈ sqrt(c) atol = 1.0e-6

# explicit inner: NewtonRaphson
s_nr = solve(mkprob(), HomotopySweep(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(s_nr)
@test s_nr.u[1] ≈ sqrt(c) atol = 1.0e-6

# explicit inner: SimpleNewtonRaphson (proves no hardcoded dependency on a specific alg)
s_snr = solve(mkprob(), HomotopySweep(; inner = SimpleNewtonRaphson()))
@test SciMLBase.successful_retcode(s_snr)
@test s_snr.u[1] ≈ sqrt(c) atol = 1.0e-6
