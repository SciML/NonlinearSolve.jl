using NonlinearSolve

using OrdinaryDiffEqTsit5
using SciMLLogging: SciMLLogging

function ode_func!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
    return nothing
end

function objective_function!(resid, u0, p)
    odeprob = ODEProblem{true}(ode_func!, u0, (0.0, 100.0), p)
    sol = solve(
        odeprob, Tsit5(), abstol = 1.0e-9, reltol = 1.0e-9, verbose = SciMLLogging.None()
    )
    resid[1] = sol(0.0)[1]
    resid[2] = sol(100.0)[1] - 1.0
    return nothing
end

prob = NonlinearProblem{true}(objective_function!, [0.0, 1.0] .+ 1im)
# Use NewtonRaphson with AutoFiniteDiff since:
# 1. ForwardDiff doesn't support complex numbers
# 2. Trust region methods use extrema which doesn't work with complex numbers
sol = solve(prob, NewtonRaphson(; autodiff = AutoFiniteDiff()); abstol = 1.0e-10)
@test SciMLBase.successful_retcode(sol)
# This test is not meant to return success but test that Newton-based solvers can handle
# complex valued problems
@test_nowarn solve(
    prob, NewtonRaphson(; autodiff = AutoFiniteDiff()); abstol = 1.0e-19, maxiters = 10
)
