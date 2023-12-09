# Nonlinear Least Squares Solvers

`solve(prob::NonlinearLeastSquaresProblem, alg; kwargs...)`

Solves the nonlinear least squares problem defined by `prob` using the algorithm
`alg`. If no algorithm is given, a default algorithm will be chosen.

## Recommended Methods

The default method `FastShortcutNLLSPolyalg` is a good choice for most problems. It is a
polyalgorithm that attempts to use a fast algorithm (`GaussNewton`) and if that fails it
falls back to a more robust algorithm (`LevenbergMarquardt`).

## Full List of Methods

### NonlinearSolve.jl

  - `LevenbergMarquardt()`: An advanced Levenberg-Marquardt implementation with the
    improvements suggested in the [paper](https://arxiv.org/abs/1201.5885) "Improvements to
    the Levenberg-Marquardt algorithm for nonlinear least-squares minimization". Designed
    for large-scale and numerically-difficult nonlinear systems.
  - `GaussNewton()`: An advanced GaussNewton implementation with support for efficient
    handling of sparse matrices via colored automatic differentiation and preconditioned
    linear solvers. Designed for large-scale and numerically-difficult nonlinear least
    squares problems.
  - `TrustRegion()`: A Newton Trust Region dogleg method with swappable nonlinear solvers and
    autodiff methods for high performance on large and sparse systems.

### SimpleNonlinearSolve.jl

These methods are included with NonlinearSolve.jl by default, though SimpleNonlinearSolve.jl
can be used  directly to reduce dependencies and improve load times.
SimpleNonlinearSolve.jl's methods excel at small problems and problems defined with static
arrays.

  - `SimpleGaussNewton()`: Simple Gauss Newton implementation using QR factorizations for
    numerical stability.

### FastLevenbergMarquardt.jl

A wrapper over
[FastLevenbergMarquardt.jl](https://github.com/kamesy/FastLevenbergMarquardt.jl). Note that
it is called `FastLevenbergMarquardt` since the original package is called "Fast", though
benchmarks demonstrate `LevenbergMarquardt()` usually outperforms.

  - `FastLevenbergMarquardtJL(linsolve = :cholesky)`, can also choose `linsolve = :qr`.

### LeastSquaresOptim.jl

A wrapper over
[LeastSquaresOptim.jl](https://github.com/matthieugomez/LeastSquaresOptim.jl). Has a core
algorithm `LeastSquaresOptimJL(alg; linsolve)` where the choices for `alg` are:

  - `:lm` a Levenberg-Marquardt implementation
  - `:dogleg` a trust-region dogleg Gauss-Newton

And the choices for `linsolve` are:

  - `:qr`
  - `:cholesky`
  - `:lsmr` a conjugate gradient method (LSMR with diagonal preconditioner).

### MINPACK.jl

MINPACK.jl methods are fine for medium-sized nonlinear solves. They are the FORTRAN
standard methods which are used in many places, such as SciPy. However, our benchmarks
demonstrate that these methods are not robust or stable. In addition, they are slower
than the standard methods and do not scale due to lack of sparse Jacobian support.
Thus they are only recommended for benchmarking and testing code conversions.

  - `CMINPACK()`: A wrapper for using the classic MINPACK method through [MINPACK.jl](https://github.com/sglyon/MINPACK.jl)

Submethod choices for this algorithm include:

  - `:hybr`: Modified version of Powell's algorithm.
  - `:lm`: Levenberg-Marquardt.
  - `:lmdif`: Advanced Levenberg-Marquardt
  - `:hybrd`: Advanced modified version of Powell's algorithm

### Optimization.jl

`NonlinearLeastSquaresProblem`s can be converted into an `OptimizationProblem`  and used
with any solver of [Optimization.jl](https://github.com/SciML/Optimization.jl).
