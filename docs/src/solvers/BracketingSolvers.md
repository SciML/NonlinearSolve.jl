# Interval Rootfinding Methods (Bracketing Solvers)

`solve(prob::IntervalNonlinearProblem,alg;kwargs)`

Solves for ``f(t)=0`` in the problem defined by `prob` using the algorithm
`alg`. If no algorithm is given, a default algorithm will be chosen.

## Recommended Methods

`Falsi()` can have a faster convergence and is discretely differentiable, but is
less stable than `Bisection`.
`Ridder` is a hybrid method that uses the value of function at the midpoint of the interval to perform an exponential interpolation to the root. This gives a fast convergence with a guaranteed convergence of at most twice the number of iterations as the bisection method.
`Brent` is a combination of the bisection method, the secant method and inverse quadratic interpolation. At every iteration, Brent's method decides which method out of these three is likely to do best, and proceeds by doing a step according to that method. This gives a robust and fast method, which therefore enjoys considerable popularity.

## Full List of Methods

### SimpleNonlinearSolve.jl

These methods are automatically included as part of NonlinearSolve.jl. Though, one can use
SimpleNonlinearSolve.jl directly to decrease the dependencies and improve load time.

  - `Falsi`: A non-allocating regula falsi method
  - `Bisection`: A common bisection method
  - `Ridder`: A non-allocating Ridder method
  - `Brent`: A non-allocating Brent method
