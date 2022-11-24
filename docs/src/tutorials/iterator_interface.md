# Nonlinear Solver Iterator Interface

There is an iterator form of the nonlinear solver which mirrors the DiffEq integrator interface:

```julia
f(u, p) = u .* u .- 2.0
u0 = 1.5 # brackets
probB = NonlinearProblem(f, u0)
cache = init(probB, NewtonRaphson()) # Can iterate the solver object
solver = solve!(cache)
```
