using NonlinearSolve

using NaNMath

# https://github.com/SciML/NonlinearSolve.jl/issues/187
# If we use a General Nonlinear Solver the solution might go out of the domain!
ff_interval(u, p) = 0.5 / 1.5 * NaNMath.log.(u ./ (1.0 .- u)) .- 2.0 * u .+ 1.0

uspan = (0.02, 0.1)
prob = IntervalNonlinearProblem(ff_interval, uspan)
sol = solve(prob; abstol = 1.0e-9)
@test SciMLBase.successful_retcode(sol)

u0 = 0.06
p = 2.0
prob = NonlinearProblem(ff_interval, u0, p)
sol = solve(prob; abstol = 1.0e-9)
@test SciMLBase.successful_retcode(sol)
