using NonlinearSolveFirstOrder

using NonlinearSolveFirstOrder, ForwardDiff
fn_iip = NonlinearFunction{true}((du, u, p) -> du .= u .* u .- p)
u2 = [
    ForwardDiff.Dual(BigFloat(1.0), 5.0), ForwardDiff.Dual(BigFloat(1.0), 5.0),
    ForwardDiff.Dual(BigFloat(1.0), 5.0),
]
prob_iip_bf = NonlinearProblem{true}(fn_iip, u2, ForwardDiff.Dual(BigFloat(2.0), 5.0))
sol = solve(prob_iip_bf, NewtonRaphson())
@test sol.retcode == ReturnCode.Success
