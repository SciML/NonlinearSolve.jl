# [Nonlinear System Solvers](@id nonlinearsystemsolvers)

`solve(prob::NonlinearProblem,alg;kwargs)`

Solves for ``f(u)=0`` in the problem defined by `prob` using the algorithm
`alg`. If no algorithm is given, a default algorithm will be chosen.

## Recommended Methods

The default method `FastShortcutNonlinearPolyalg` is a good choice for most
problems. It is a polyalgorithm that attempts to use a fast algorithm
(Klement, Broyden) and if that fails it falls back to a more robust
algorithm (`NewtonRaphson`) before falling back the most robust varient of
`TrustRegion`. For basic problems this will be very fast, for harder problems
it will make sure to work.

If one is looking for more robustness then `RobustMultiNewton` is a good choice.
It attempts a set of the most robust methods in succession and only fails if
all of the methods fail to converge. Additionally, `DynamicSS` can be a good choice 
for high stability.

As a balance, `NewtonRaphson` is a good choice for most problems that aren't too
difficult yet need high performance, and  `TrustRegion` is a bit less performant
but more stable. If the problem is well-conditioned, `Klement` or `Broyden`
may be faster, but highly dependent on the eigenvalues of the Jacobian being
sufficiently small.

`NewtonRaphson` and `TrustRegion` are designed for for large systems. 
They can make use of sparsity patterns for sparse automatic differentiation
and sparse linear solving of very large systems. Meanwhile,
`SimpleNewtonRaphson` and `SimpleTrustRegion` are implementations which is specialized for
small equations. They are non-allocating on static arrays and thus really well-optimized
for small systems, thus usually outperforming the other methods when such types are
used for `u0`. 

## Full List of Methods

!!! note
    
    For the full details on the capabilities and constructors of the different solvers,
    see the Detailed Solver APIs section!

### NonlinearSolve.jl

These are the core solvers, which excel at large-scale problems that need advanced
linear solver, automatic differentiation, abstract array types, GPU,
sparse/structured matrix support, etc. These methods support the largest set of types and
features, but have a bit of overhead on very small problems.

  - `NewtonRaphson()`:A Newton-Raphson method with swappable nonlinear solvers and autodiff
    methods for high performance on large and sparse systems.
  - `TrustRegion()`: A Newton Trust Region dogleg method with swappable nonlinear solvers and
    autodiff methods for high performance on large and sparse systems.
  - `LevenbergMarquardt()`: An advanced Levenberg-Marquardt implementation with the
    improvements suggested in the [paper](https://arxiv.org/abs/1201.5885) "Improvements to
    the Levenberg-Marquardt algorithm for nonlinear least-squares minimization". Designed for
    large-scale and numerically-difficult nonlinear systems.
  - `RobustMultiNewton()`: A polyalgorithm that mixes highly robust methods (line searches and
    trust regions) in order to be as robust as possible for difficult problems. If this method
    fails to converge, then one can be pretty certain that most (all?) other choices would 
    likely fail.
  - `FastShortcutNonlinearPolyalg`: The default method. A polyalgorithm that mixes fast methods
    with fallbacks to robust methods to allow for solving easy problems quickly without sacrificing
    robustnes on the hard problems.

### SimpleNonlinearSolve.jl

These methods are included with NonlinearSolve.jl by default, though SimpleNonlinearSolve.jl
can be used directly to reduce dependencies and improve load times. SimpleNonlinearSolve.jl's
methods excel at small problems and problems defined with static arrays.

  - `SimpleNewtonRaphson()`: A simplified implementation of the Newton-Raphson method.
  - `Broyden()`: The classic Broyden's quasi-Newton method.
  - `Klement()`: A quasi-Newton method due to Klement. It's supposed to be more efficient
    than Broyden's method, and it seems to be in the cases that have been tried, but more
    benchmarking is required.
  - `SimpleTrustRegion()`: A dogleg trust-region Newton method. Improved globalizing stability
    for more robust fitting over basic Newton methods, though potentially with a cost.
  - `SimpleDFSane()`: A low-overhead implementation of the df-sane method for solving
    large-scale nonlinear systems of equations.

!!! note
    
    When used with certain types for the states `u` such as a `Number` or `StaticArray`,
    these solvers are very efficient and non-allocating. These implementations are thus
    well-suited for small systems of equations.

### SteadyStateDiffEq.jl

SteadyStateDiffEq.jl uses ODE solvers to iteratively approach the steady state. It is a
very stable method for solving nonlinear systems, though often more
computationally expensive than direct methods.

  - `DynamicSS()` : Uses an ODE solver to find the steady state. Automatically
    terminates when close to the steady state.

### SciMLNLSolve.jl

This is a wrapper package for importing solvers from NLsolve.jl into the SciML interface.

  - `NLSolveJL()`: A wrapper for [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl)

Submethod choices for this algorithm include:

  - `:fixedpoint`: Fixed-point iteration
  - `:anderson`: Anderson-accelerated fixed-point iteration
  - `:newton`: Classical Newton method with an optional line search
  - `:trust_region`: Trust region Newton method (the default choice)

### MINPACK.jl

MINPACK.jl methods are good for medium-sized nonlinear solves. It does not scale due to
the lack of sparse Jacobian support, though the methods are very robust and stable.

  - `CMINPACK()`: A wrapper for using the classic MINPACK method through [MINPACK.jl](https://github.com/sglyon/MINPACK.jl)

Submethod choices for this algorithm include:

  - `:hybr`: Modified version of Powell's algorithm.
  - `:lm`: Levenberg-Marquardt.
  - `:lmdif`: Advanced Levenberg-Marquardt
  - `:hybrd`: Advanced modified version of Powell's algorithm

### Sundials.jl

Sundials.jl are a classic set of C/Fortran methods which are known for good scaling of the
Newton-Krylov form. However, KINSOL is known to be less stable than some other
implementations, as it has no line search or globalizer (trust region).

  - `KINSOL()`: The KINSOL method of the SUNDIALS C library
