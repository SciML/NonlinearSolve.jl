# [Interval Root-Finding Methods (Bracketing Solvers)](@id bracketing)

```julia
solve(prob::IntervalNonlinearProblem, alg; kwargs...)
```

Solves for ``f(t) = 0`` in the problem defined by `prob` using the algorithm `alg`. If no
algorithm is given, a default algorithm will be chosen.

## Recommended Methods

[`modAB`](@ref) (Modified Anderson-Bjork) is the recommended method for the scalar interval root-finding problems. It combines Bisection with Anderson-Bjork steps to achieve superlinear convergence 0.7 ÷ 0.8, providing optimal convergence rate for poorly behaved functions. According to our benchmarks, it outperforms the other methods in most cases.

[`ITP`](@ref) is particularly well-suited for cases where the function is smooth and well-behaved; and
achieved superlinear convergence while retaining the optimal worst-case performance of the
Bisection method. For more details, consult the detailed solver API docs.

[`Ridder`](@ref) is a hybrid method that uses the value of function at the midpoint of the
interval to perform an exponential interpolation to the root. This gives a fast convergence
with a guaranteed convergence of at most twice the number of iterations as the bisection
method.

[`Brent`](@ref) is a combination of the bisection method, the secant method and inverse
quadratic interpolation. At every iteration, Brent's method decides which method out of
these three is likely to do best, and proceeds by doing a step according to that method.
This gives a robust and fast method, which therefore enjoys considerable popularity.

## Full List of Methods

### BracketingNonlinearSolve.jl

These methods are automatically included as part of NonlinearSolve.jl. Though, one can use
BracketingNonlinearSolve.jl directly to decrease the dependencies and improve load time.

  - [`Alefeld`](@ref): A non-allocating Alefeld method
  - [`Bisection`](@ref): A common bisection method
  - [`Brent`](@ref): A non-allocating Brent method
  - [`Falsi`](@ref): A non-allocating regula falsi method
  - [`ITP`](@ref): A non-allocating ITP (Interpolate, Truncate & Project) method
  - [`ModAB`](@ref): A non-allocating Modified Anderson-Bjork's method
  - [`Muller`](@ref): A non-allocating Muller's method
  - [`Ridder`](@ref): A non-allocating Ridder method
