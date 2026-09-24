using NonlinearSolveFirstOrder, LinearAlgebra, SciMLBase

prob = NonlinearLeastSquaresProblem(
    (u, p) -> [u[1] + u[2] - 0.6, 0.0, 0.0], zeros(2)
)
sol = solve(prob, NewtonRaphson(); abstol = 1.0e-10)
@test SciMLBase.successful_retcode(sol)
@test norm(sol.resid) < 1.0e-8
