# Nonlinear Solver Iterator Interface

There is an iterator form of the nonlinear solver which mirrors the DiffEq integrator interface:

```julia
f(u, p) = u .* u .- 2.0
u0 = (1.0, 2.0) # brackets
probB = NonlinearProblem(f, u0)
solver = init(probB, Falsi()) # Can iterate the solver object
solver = solve!(solver)
```

Note that the `solver` object is actually immutable since we want to make it
live on the stack for the sake of performance.
