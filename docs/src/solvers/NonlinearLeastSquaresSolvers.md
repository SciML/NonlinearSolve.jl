# Nonlinear Least Squares Solvers

`solve(prob::NonlinearLeastSquaresProblem, alg; kwargs...)`

Solves the nonlinear least squares problem defined by `prob` using the algorithm
`alg`. If no algorithm is given, a default algorithm will be chosen.

## Recommended Methods

The default method `FastShortcutNLLSPolyalg` is a good choice for most
problems. It is a polyalgorithm that attempts to use a fast algorithm
(`GaussNewton`) and if that fails it falls back to a more robust
algorithm (`LevenbergMarquardt`).

## Full List of Methods

  - `LevenbergMarquardt()`: An advanced Levenberg-Marquardt implementation with the
    improvements suggested in the [paper](https://arxiv.org/abs/1201.5885) "Improvements to
    the Levenberg-Marquardt algorithm for nonlinear least-squares minimization". Designed for
    large-scale and numerically-difficult nonlinear systems.
  - `GaussNewton()`: An advanced GaussNewton implementation with support for efficient
    handling of sparse matrices via colored automatic differentiation and preconditioned
    linear solvers. Designed for large-scale and numerically-difficult nonlinear least squares
    problems.
  - `SimpleNewtonRaphson()`: Simple Gauss Newton Implementation with `QRFactorization` to
    solve a linear least squares problem at each step!
