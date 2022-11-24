# Interval Rootfinding Methods (Bracketing Solvers)

`solve(prob::IntervalNonlinearProblem,alg;kwargs)`

Solves for ``f(t)=0`` in the problem defined by `prob` using the algorithm
`alg`. If no algorithm is given, a default algorithm will be chosen.

This page is solely focused on the bracketing methods for scalar nonlinear equations.

## Recommended Methods

`Falsi()` can have a faster convergence and is discretely differentiable, but is
less stable than `Bisection`.

## Full List of Methods

### SimpleNonlinearSolve.jl

These methods are automatically included as part of NonlinearSolve.jl. Though one can use
SimpleNonlinearSolve.jl directly to decrease the dependencies and improve load time.

- `Falsi`: A non-allocating regula falsi method
- `Bisection`: A common bisection method
