using SimpleNonlinearSolve, SciMLBase

# `precondition` composes through the SimpleNonlinearSolve bypass `solve` entry: the
# asinh-compressed diode residual converges where plain Newton needs ~60 creeping steps.
p = (; Is = 1.0e-14, Vt = 0.025, It = 1.0e-2)
f_diode = (v, p) -> p.Is * expm1(v / p.Vt) - p.It
vstar = p.Vt * log(p.It / p.Is + 1)
G = (fu, u, p) -> asinh(fu)

sol_plain = solve(NonlinearProblem(f_diode, 2.0, p), SimpleNewtonRaphson(); maxiters = 15)
@test !SciMLBase.successful_retcode(sol_plain)

prob_G = NonlinearProblem(NonlinearFunction(f_diode; precondition = G), 2.0, p)
sol_G = solve(prob_G, SimpleNewtonRaphson(); maxiters = 15)
@test SciMLBase.successful_retcode(sol_G)
@test abs(sol_G.u - vstar) < 1.0e-8

# `postcondition` has no iterate-commit support in the simple solvers and must error
# rather than being silently ignored.
H = (up, uprev, p) -> up
prob_H = NonlinearProblem(NonlinearFunction(f_diode; postcondition = H), 2.0, p)
@test_throws ArgumentError solve(prob_H, SimpleNewtonRaphson())
