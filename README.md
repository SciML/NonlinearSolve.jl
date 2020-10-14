# NonlinearSolve.jl

Fast implementations of root finding algorithms in Julia that satisfy the SciML common interface.

```julia
using NonlinearSolve, StaticArrays

f(u,p) = u .* u .- 2
u0 = @SVector[1.0, 1.0]
probN = NonlinearProblem{false}(f, u0)
solver = solve(probN, NewtonRaphson(), tol = 1e-9)

## Bracketing Methods

f(u, p) = u .* u .- 2.0
u0 = (1.0, 2.0) # brackets
probB = NonlinearProblem(f, u0)
sol = solve(probB, Falsi())
```

## Current Algorithms 

### Non-Bracketing

- `NewtonRaphson()`

### Bracketing

- `Falsi()`
- `Bisection()`

## Features

Performance is key: the current methods are made to be highly performant on scalar and statically sized small
problems. If you run into any performance issues, please file an issue.

There is an iterator form of the nonlinear solver which mirrors the DiffEq integrator interface:

```julia
f(u, p) = u .* u .- 2.0
u0 = (1.0, 2.0) # brackets
probB = NonlinearProblem(f, u0)
solver = init(probB, Falsi()) # Can iterate the solver object
solver = solve!(solver)
```

Note that the `solver` object is actually immutable since we want to make it live on the stack for the sake of performance.

## Roadmap

The current algorithms should support automatic differentiation, though improved adjoint overloads are planned
to be added in the current update (which will make use of the `f(u,p)` form). Future updates will include
standard methods for larger scale nonlinear solving like Newton-Krylov methods.
