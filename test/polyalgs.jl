using NonlinearSolve, Test, NaNMath

f(u, p) = u .* u .- 2
u0 = [1.0, 1.0]
probN = NonlinearProblem{false}(f, u0)

custom_polyalg = NonlinearSolvePolyAlgorithm((GeneralBroyden(), LimitedMemoryBroyden()))

# Uses the `__solve` function
@time solver = solve(probN; abstol = 1e-9)
@test SciMLBase.successful_retcode(solver)
@time solver = solve(probN, RobustMultiNewton(); abstol = 1e-9)
@test SciMLBase.successful_retcode(solver)
@time solver = solve(probN, FastShortcutNonlinearPolyalg(); abstol = 1e-9)
@test SciMLBase.successful_retcode(solver)
@time solver = solve(probN, custom_polyalg; abstol = 1e-9)
@test SciMLBase.successful_retcode(solver)

# Test the caching interface
cache = init(probN; abstol = 1e-9);
@time solver = solve!(cache)
@test SciMLBase.successful_retcode(solver)
cache = init(probN, RobustMultiNewton(); abstol = 1e-9);
@time solver = solve!(cache)
@test SciMLBase.successful_retcode(solver)
cache = init(probN, FastShortcutNonlinearPolyalg(); abstol = 1e-9);
@time solver = solve!(cache)
@test SciMLBase.successful_retcode(solver)
cache = init(probN, custom_polyalg; abstol = 1e-9);
@time solver = solve!(cache)
@test SciMLBase.successful_retcode(solver)

# https://github.com/SciML/NonlinearSolve.jl/issues/153
function f(du, u, p)
    s1, s1s2, s2 = u
    k1, c1, Δt = p

    du[1] = -0.25 * c1 * k1 * s1 * s2
    du[2] = 0.25 * c1 * k1 * s1 * s2
    du[3] = -0.25 * c1 * k1 * s1 * s2
end

prob = NonlinearProblem(f, [2.0, 2.0, 2.0], [1.0, 2.0, 2.5])
sol = solve(prob; abstol = 1e-9)
@test SciMLBase.successful_retcode(sol)

# https://github.com/SciML/NonlinearSolve.jl/issues/187
# If we use a General Nonlinear Solver the solution might go out of the domain!
ff(u, p) = 0.5 / 1.5 * NaNMath.log.(u ./ (1.0 .- u)) .- 2.0 * u .+ 1.0

uspan = (0.02, 0.1)
prob = IntervalNonlinearProblem(ff, uspan)
sol = solve(prob; abstol = 1e-9)
@test SciMLBase.successful_retcode(sol)

u0 = 0.06
p = 2.0
prob = NonlinearProblem(ff, u0, p)
sol = solve(prob; abstol = 1e-9)
@test SciMLBase.successful_retcode(sol)
