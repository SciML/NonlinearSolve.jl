using NonlinearSolve

f(u, p) = u .* u .- 2
u0 = [1.0, 1.0]
probN = NonlinearProblem(f, u0)
@time solver = solve(probN, abstol = 1e-9)
@time solver = solve(probN, RobustMultiNewton(), abstol = 1e-9)
@time solver = solve(probN, FastShortcutNonlinearPolyalg(), abstol = 1e-9)