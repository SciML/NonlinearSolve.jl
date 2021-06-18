# Bracketing Solvers

`solve(prob::NonlinearProblem,alg;kwargs)`

Solves for ``f(u)=0`` in the problem defined by `prob` using the algorithm
`alg`. If no algorithm is given, a default algorithm will be chosen.

This page is solely focused on the bracketing methods for scalar nonlinear equations.

## Recommended Methods

`Falsi()` can have a faster convergence and is discretely differentiable, but is
less stable than `Bisection`.

## Full List of Methods

### NonlinearSolve.jl

- `Falsi` : A non-allocating regula falsi method
- `Bisection`: A common bisection method
