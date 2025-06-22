# [Nonlinear System Solvers](@id nonlinearsystemsolvers)

```julia
solve(prob::NonlinearProblem, alg; kwargs...)
```

Solves for ``f(u) = 0`` in the problem defined by `prob` using the algorithm `alg`. If no
algorithm is given, a default algorithm will be chosen.

## Recommended Methods

The default method [`FastShortcutNonlinearPolyalg`](@ref) is a good choice for most
problems. It is a polyalgorithm that attempts to use a fast algorithm ([`Klement`](@ref),
[`Broyden`](@ref)) and if that fails it falls back to a more robust algorithm
([`NewtonRaphson`](@ref)) before falling back the most robust variant of
[`TrustRegion`](@ref). For basic problems this will be very fast, for harder problems it
will make sure to work.

If one is looking for more robustness then [`RobustMultiNewton`](@ref) is a good choice. It
attempts a set of the most robust methods in succession and only fails if all of the methods
fail to converge. Additionally, [`DynamicSS`](@ref) can be a good choice for high stability
if the root corresponds to a stable equilibrium.

As a balance, [`NewtonRaphson`](@ref) is a good choice for most problems that aren't too
difficult yet need high performance, and  [`TrustRegion`](@ref) is a bit less performant but
more stable. If the problem is well-conditioned, [`Klement`](@ref) or [`Broyden`](@ref) may
be faster, but highly dependent on the eigenvalues of the Jacobian being sufficiently small.

[`NewtonRaphson`](@ref) and [`TrustRegion`](@ref) are designed for for large systems. They
can make use of sparsity patterns for sparse automatic differentiation and sparse linear
solving of very large systems. Meanwhile, [`SimpleNewtonRaphson`](@ref) and
[`SimpleTrustRegion`](@ref) are implementations which are specialized for small equations.
They are non-allocating on static arrays and thus really well-optimized for small systems,
thus usually outperforming the other methods when such types are used for `u0`.
Additionally, these solvers can be used inside GPU kernels. See
[ParallelParticleSwarms.jl](https://github.com/SciML/ParallelParticleSwarms.jl) for an example of this.

## Full List of Methods

!!! note
    
    For the full details on the capabilities and constructors of the different solvers,
    see the Detailed Solver APIs section!

### NonlinearSolve.jl

These are the core solvers, which excel at large-scale problems that need advanced
linear solver, automatic differentiation, abstract array types, GPU,
sparse/structured matrix support, etc. These methods support the largest set of types and
features, but have a bit of overhead on very small problems.

  - [`NewtonRaphson()`](@ref): A Newton-Raphson method with swappable nonlinear solvers and
    autodiff methods for high performance on large and sparse systems.
  - [`TrustRegion()`](@ref): A Newton Trust Region dogleg method with swappable nonlinear
    solvers and autodiff methods for high performance on large and sparse systems.
  - [`LevenbergMarquardt()`](@ref): An advanced Levenberg-Marquardt implementation with the
    improvements suggested in the [transtrum2012improvements](@citet). Designed for
    large-scale and numerically-difficult nonlinear systems.
  - [`PseudoTransient()`](@ref): A pseudo-transient method which mixes the stability of
    Euler-type stepping with the convergence speed of a Newton method. Good for highly
    unstable systems.
  - [`RobustMultiNewton()`](@ref): A polyalgorithm that mixes highly robust methods (line
    searches and trust regions) in order to be as robust as possible for difficult problems.
    If this method fails to converge, then one can be pretty certain that most (all?) other
    choices would likely fail.
  - [`FastShortcutNonlinearPolyalg()`](@ref): The default method. A polyalgorithm that mixes
    fast methods with fallbacks to robust methods to allow for solving easy problems quickly
    without sacrificing robustness on the hard problems.
  - [`Broyden()`](@ref): Generalization of Broyden's Quasi-Newton Method with Line Search
    and Automatic Jacobian Resetting. This is a fast method but unstable when the condition
    number of the Jacobian matrix is sufficiently large.
  - [`Klement()`](@ref): Generalization of Klement's Quasi-Newton Method with Line Search
    and Automatic Jacobian Resetting. This is a fast method but unstable when the condition
    number of the Jacobian matrix is sufficiently large.
  - [`LimitedMemoryBroyden()`](@ref): An advanced version of
    [`SimpleLimitedMemoryBroyden`](@ref) which uses a limited memory Broyden method. This is
    a fast method but unstable when the condition number of the Jacobian matrix is
    sufficiently large. It is recommended to use [`Broyden`](@ref) or [`Klement`](@ref)
    instead unless the memory usage is a concern.

### SimpleNonlinearSolve.jl

These methods are included with NonlinearSolve.jl by default, though SimpleNonlinearSolve.jl
can be used directly to reduce dependencies and improve load times.
SimpleNonlinearSolve.jl's methods excel at small problems and problems defined with static
arrays.

  - [`SimpleNewtonRaphson()`](@ref): A simplified implementation of the Newton-Raphson
    method.
  - [`SimpleBroyden()`](@ref): The classic Broyden's quasi-Newton method.
  - [`SimpleLimitedMemoryBroyden()`](@ref): A low-memory Broyden implementation, similar to
    L-BFGS. This method is common in machine learning contexts but is known to be unstable
    in comparison to many other choices.
  - [`SimpleKlement()`](@ref): A quasi-Newton method due to Klement. It's supposed to be
    more efficient than Broyden's method, and it seems to be in the cases that have been
    tried, but more benchmarking is required.
  - [`SimpleTrustRegion()`](@ref): A dogleg trust-region Newton method. Improved globalizing
    stability for more robust fitting over basic Newton methods, though potentially with a
    cost.
  - [`SimpleDFSane()`](@ref): A low-overhead implementation of the df-sane method for
    solving large-scale nonlinear systems of equations.
  - [`SimpleHalley()`](@ref): A low-overhead implementation of the Halley method. This is a
    higher order method and thus can converge faster to low tolerances than a Newton method.
    Requires higher order derivatives, so best used when automatic differentiation is
    available.

!!! note
    
    When used with certain types for the states `u` such as a `Number` or `StaticArray`,
    these solvers are very efficient and non-allocating. These implementations are thus
    well-suited for small systems of equations.

### SteadyStateDiffEq.jl

SteadyStateDiffEq.jl uses ODE solvers to iteratively approach the steady state. It is a
very stable method for solving nonlinear systems with stable equilibrium points, though
often more computationally expensive than direct methods.

  - [`DynamicSS()`](@ref): Uses an ODE solver to find the steady state. Automatically
    terminates when close to the steady state.
  - [`SSRootfind()`](@ref): Uses a NonlinearSolve compatible solver to find the steady
    state.

### NLsolve.jl

This is a wrapper package for importing solvers from NLsolve.jl into the SciML interface.

  - [`NLsolveJL()`](@ref): A wrapper for
    [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl)

Submethod choices for this algorithm include:

  - `:anderson`: Anderson-accelerated fixed-point iteration
  - `:newton`: Classical Newton method with an optional line search
  - `:trust_region`: Trust region Newton method (the default choice)

### MINPACK.jl

MINPACK.jl is a wrapper package for bringing the Fortran solvers from MINPACK. However, our
benchmarks reveal that these methods are rarely competitive with our native solvers. Thus,
our recommendation is to use these only for benchmarking and debugging purposes.

  - [`CMINPACK()`](@ref): A wrapper for using the classic MINPACK method through
    [MINPACK.jl](https://github.com/sglyon/MINPACK.jl)

Submethod choices for this algorithm include:

  - `:hybr`: Modified version of Powell's algorithm.
  - `:lm`: Levenberg-Marquardt.
  - `:lmdif`: Advanced Levenberg-Marquardt
  - `:hybrd`: Advanced modified version of Powell's algorithm

### Sundials.jl

Sundials.jl are a classic set of C/Fortran methods which are known for good scaling of the
Newton-Krylov form. However, KINSOL is known to be less stable than some other
implementations.

  - [`KINSOL()`](@ref): The KINSOL method of the SUNDIALS C library

### SIAMFANLEquations.jl

SIAMFANLEquations.jl is a wrapper for the methods in the SIAMFANLEquations.jl library.

  - [`SIAMFANLEquationsJL()`](@ref): A wrapper for using the methods in
    [SIAMFANLEquations.jl](https://github.com/ctkelley/SIAMFANLEquations.jl)

Other solvers listed in [Fixed Point Solvers](@ref fixed_point_methods_full_list),
[FastLevenbergMarquardt.jl](@ref fastlm_wrapper_summary) and
[LeastSquaresOptim.jl](@ref lso_wrapper_summary) can also solve nonlinear systems.

### NLSolvers.jl

This is a wrapper package for importing solvers from NLSolvers.jl into the SciML interface.

  - [`NLSolversJL()`](@ref): A wrapper for
    [NLSolvers.jl](https://github.com/JuliaNLSolvers/NLSolvers.jl)

For a list of possible solvers see the [NLSolvers.jl documentation](https://julianlsolvers.github.io/NLSolvers.jl/)

### PETSc.jl

This is a wrapper package for importing solvers from PETSc.jl into the SciML interface.

  - [`PETScSNES()`](@ref): A wrapper for
    [PETSc.jl](https://github.com/JuliaParallel/PETSc.jl)

For a list of possible solvers see the [PETSc.jl documentation](https://petsc.org/release/manual/snes/)

### SciPy (Python via PythonCall)

These wrappers let you use the algorithms from
[`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html)
without leaving Julia.  SciPy is loaded lazily through PythonCall, so these
methods are available whenever the `scipy` Python package can be imported.

  - [`SciPyRoot()`](@ref): wrapper for `scipy.optimize.root` (vector problems)
  - [`SciPyRootScalar()`](@ref): wrapper for `scipy.optimize.root_scalar` (scalar/bracketed problems)
