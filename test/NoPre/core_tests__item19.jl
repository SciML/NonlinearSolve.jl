using NonlinearSolve

f(u, p) = [u[1]^2 - 2u[1] + 1, sum(u)]
prob = NonlinearProblem(f, [1.0, 1.0])
sol = solve(prob)
@test SciMLBase.successful_retcode(sol)
