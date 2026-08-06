using SimpleNonlinearSolve, SciMLBase
import NonlinearSolveBase
using SciMLLogging: SciMLLogging

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

# `postcondition` has no iterate-commit support in the simple solvers, so it is reported
# at ErrorLevel rather than silently ignored — and, being a verbosity toggle, it can be
# turned down when that is deliberate.
@test_throws ErrorException solve(
    prob, SimpleNewtonRaphson(); postcondition = (up, uprev, p, cache) -> up
)
sol_silenced = solve(
    prob, SimpleNewtonRaphson(); postcondition = (up, uprev, p, cache) -> up,
    verbose = NonlinearSolveBase.NonlinearVerbosity(SciMLLogging.None()), maxiters = 200
)
@test SciMLBase.successful_retcode(sol_silenced)
