using NonlinearSolve, Test, NaNMath, OrdinaryDiffEq

f(u, p) = u .* u .- 2
u0 = [1.0, 1.0]
probN = NonlinearProblem{false}(f, u0)

custom_polyalg = NonlinearSolvePolyAlgorithm((Broyden(), LimitedMemoryBroyden()))

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
ff_interval(u, p) = 0.5 / 1.5 * NaNMath.log.(u ./ (1.0 .- u)) .- 2.0 * u .+ 1.0

uspan = (0.02, 0.1)
prob = IntervalNonlinearProblem(ff_interval, uspan)
sol = solve(prob; abstol = 1e-9)
@test SciMLBase.successful_retcode(sol)

u0 = 0.06
p = 2.0
prob = NonlinearProblem(ff_interval, u0, p)
sol = solve(prob; abstol = 1e-9)
@test SciMLBase.successful_retcode(sol)

# Shooting Problem: Taken from BoundaryValueDiffEq.jl
# Testing for Complex Valued Root Finding. For Complex valued inputs we drop some of the
# algorithms which dont support those.
function ode_func!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
    return nothing
end

function objective_function!(resid, u0, p)
    odeprob = ODEProblem{true}(ode_func!, u0, (0.0, 100.0), p)
    sol = solve(odeprob, Tsit5(), abstol = 1e-9, reltol = 1e-9, verbose = false)
    resid[1] = sol(0.0)[1]
    resid[2] = sol(100.0)[1] - 1.0
    return nothing
end

prob = NonlinearProblem{true}(objective_function!, [0.0, 1.0] .+ 1im)
sol = solve(prob; abstol = 1e-10)
@test SciMLBase.successful_retcode(sol)
# This test is not meant to return success but test that all the default solvers can handle
# complex valued problems
@test_nowarn solve(prob; abstol = 1e-19, maxiters = 10)
@test_nowarn solve(prob, RobustMultiNewton(eltype(prob.u0)); abstol = 1e-19, maxiters = 10)
