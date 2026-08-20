using NonlinearSolveQuasiNewton
using SciMLBase

f(u, p) = [u[1]^3 - 2, u[2]]
prob = NonlinearProblem(f, [1.0, 0.0])
cache = init(prob, Broyden(; max_resets = 1); abstol = 1.0e-12)
sol = solve!(cache)

@test SciMLBase.successful_retcode(sol)
@test maximum(abs, f(sol.u, nothing)) ≤ 1.0e-12
