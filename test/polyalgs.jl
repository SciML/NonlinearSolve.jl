using NonlinearSolve, Test

f(u, p) = u .* u .- 2
u0 = [1.0, 1.0]
probN = NonlinearProblem(f, u0)
@time solver = solve(probN, abstol = 1e-9)
@time solver = solve(probN, RobustMultiNewton(), abstol = 1e-9)
@time solver = solve(probN, FastShortcutNonlinearPolyalg(), abstol = 1e-9)

# https://github.com/SciML/NonlinearSolve.jl/issues/153

function f(du, u, p)
    s1, s1s2, s2 = u
    k1, c1, Δt = p

    du[1] = -0.25 * c1 * k1 * s1 * s2
    du[2] = 0.25 * c1 * k1 * s1 * s2
    du[3] = -0.25 * c1 * k1 * s1 * s2
end

prob = NonlinearProblem(f, [2.0,2.0,2.0], [1.0, 2.0, 2.5])
sol = solve(prob)
@test SciMLBase.successful_retcode(sol)

# https://github.com/SciML/NonlinearSolve.jl/issues/187

f(u, p) = 0.5/1.5*log.(u./(1.0.-u)) .- 2.0*u .+1.0

uspan = (0.02, 0.1)
prob = IntervalNonlinearProblem(f, uspan)
sol = solve(prob)
@test SciMLBase.successful_retcode(sol)

u0 = 0.06
p = 2.0
prob = NonlinearProblem(f, u0, p)
solver = solve(prob)
@test SciMLBase.successful_retcode(sol)