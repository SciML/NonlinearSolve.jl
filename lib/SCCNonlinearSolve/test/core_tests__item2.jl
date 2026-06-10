using SCCNonlinearSolve
include("setup_corerootfindtesting.jl")

# Regression test for https://github.com/SciML/NonlinearSolve.jl/issues/758
# SCCNonlinearProblem does not have a u0 field, so calling solve() without
# explicit u0 should not try to access prob.u0
using SciMLBase
using SCCNonlinearSolve
import NonlinearSolve

# Create simple nonlinear subproblems (OOP style, returning vectors)
f1(u, p) = [u[1]^2 - 2.0]
f2(u, p) = [u[1] - 1.0]

prob1 = NonlinearProblem(f1, [1.0])
prob2 = NonlinearProblem(f2, [1.0])

# Create explicit functions (identity - just pass through)
explicitfun1!(p, sols) = nothing
explicitfun2!(p, sols) = nothing

# Create the SCC problem using the same pattern as existing tests
scc_prob = SciMLBase.SCCNonlinearProblem(
    (prob1, prob2),
    SciMLBase.Void{Any}.([explicitfun1!, explicitfun2!])
)

# This should not throw an error about prob.u0 field
sol = SciMLBase.solve(scc_prob)

@test SciMLBase.successful_retcode(sol)
# First subproblem: u^2 = 2 → u = sqrt(2)
@test sol.u[1] ≈ sqrt(2.0)
# Second subproblem: u = 1
@test sol.u[2] ≈ 1.0
