using SimpleNonlinearSolve, SciMLBase

# `precondition` is a problem transformation, so it composes even on the bypass `solve`
# entries of the simple solvers: plain Newton needs ~60 creeping steps from v0 = 2.
p = (; Is = 1.0e-14, Vt = 0.025, It = 1.0e-2)
f_diode = (v, p) -> p.Is * expm1(v / p.Vt) - p.It
vstar = p.Vt * log(p.It / p.Is + 1)
prob = NonlinearProblem(f_diode, 2.0, p)

@test !SciMLBase.successful_retcode(solve(prob, SimpleNewtonRaphson(); maxiters = 15))
sol = solve(
    prob, SimpleNewtonRaphson(); precondition = (fu, u, p) -> asinh(fu), maxiters = 15
)
@test SciMLBase.successful_retcode(sol)
@test abs(sol.u - vstar) < 1.0e-8

# `postcondition` has no iterate-commit support in the simple solvers, so it must error
# rather than be silently ignored.
@test_throws ArgumentError solve(
    prob, SimpleNewtonRaphson(); postcondition = (up, uprev, p) -> up
)
