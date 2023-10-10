# Nonlinear Least Squares Solvers

`solve(prob::NonlinearLeastSquaresProblem, alg; kwargs...)`

Solves the nonlinear least squares problem defined by `prob` using the algorithm
`alg`. If no algorithm is given, a default algorithm will be chosen.

## Recommended Methods

`LevenbergMarquardt` is a good choice for most problems.

## Full List of Methods

  - `LevenbergMarquardt()`: An advanced Levenberg-Marquardt implementation with the
    improvements suggested in the [paper](https://arxiv.org/abs/1201.5885) "Improvements to
    the Levenberg-Marquardt algorithm for nonlinear least-squares minimization". Designed for
    large-scale and numerically-difficult nonlinear systems.
  - `GaussNewton()`: An advanced GaussNewton implementation with support for efficient
    handling of sparse matrices via colored automatic differentiation and preconditioned
    linear solvers. Designed for large-scale and numerically-difficult nonlinear least squares
    problems.
  - `SimpleNewtonRaphson()`: Newton Raphson implementation that uses Linear Least Squares
    solution at every step to compute the descent direction. **WARNING**: This method is not
    a robust solver for nonlinear least squares problems. The computed delta step might not
    be the correct descent direction!

## Example usage

```julia
using NonlinearSolve
sol = solve(prob, LevenbergMarquardt())
```
