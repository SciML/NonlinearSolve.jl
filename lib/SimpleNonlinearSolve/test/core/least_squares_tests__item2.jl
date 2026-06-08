using SimpleNonlinearSolve

using LinearAlgebra, SciMLBase

# OOP: unsatisfiable residual (issue #842)
f_unsat(u, p) = [1.0]
unsat_f = NonlinearFunction{false}(f_unsat; resid_prototype = zeros(1))
prob = NonlinearLeastSquaresProblem(unsat_f, nothing)
sol = solve(prob, SimpleNewtonRaphson())
@test sol.retcode == ReturnCode.Failure
@test sol.resid == [1.0]
@test sol.u == Float64[]

# OOP: satisfiable residual
f_sat(u, p) = [0.0]
sat_f = NonlinearFunction{false}(f_sat; resid_prototype = zeros(1))
prob_sat = NonlinearLeastSquaresProblem(sat_f, nothing)
sol_sat = solve(prob_sat, SimpleNewtonRaphson())
@test sol_sat.retcode == ReturnCode.Success
@test sol_sat.resid == [0.0]

# IIP: unsatisfiable residual
function f_unsat!(du, u, p)
    du[1] = 1.0
    return
end
unsat_f_iip = NonlinearFunction{true}(f_unsat!; resid_prototype = zeros(1))
prob_iip = NonlinearLeastSquaresProblem(unsat_f_iip, nothing)
sol_iip = solve(prob_iip, SimpleNewtonRaphson())
@test sol_iip.retcode == ReturnCode.Failure
@test sol_iip.resid == [1.0]
