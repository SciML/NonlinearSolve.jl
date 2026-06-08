using NonlinearSolve

f(u, p) = [1.0]
nlf = NonlinearFunction(f; resid_prototype = zeros(1))
prob = NonlinearLeastSquaresProblem(nlf, [1.0])
sol = solve(prob)
@test SciMLBase.successful_retcode(sol)
@test sol.retcode == ReturnCode.StalledSuccess
