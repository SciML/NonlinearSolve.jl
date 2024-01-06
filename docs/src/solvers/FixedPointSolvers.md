# Fixed Point Solvers

Currently we don't have an API to directly specify Fixed Point Solvers. However, a Fixed
Point Problem can be trivially converted to a Root Finding Problem. Say we want to solve:

```math
f(u) = u
```

This can be written as:

```math
g(u) = f(u) - u = 0
```

``g(u) = 0`` is a root finding problem. Note that we can use any root finding
algorithm to solve this problem. However, this is often not the most efficient way to
solve a fixed point problem. We provide a few algorithms available via extensions that
are more efficient for fixed point problems.

Note that even if you use one of the Fixed Point Solvers mentioned here, you must still
use the `NonlinearProblem` API to specify the problem, i.e., ``g(u) = 0``.

## Recommended Methods

Using [native NonlinearSolve.jl methods](@ref nonlinearsystemsolvers) is the recommended
approach. For systems where constructing Jacobian Matrices are expensive, we recommend
using a Krylov Method with one of those solvers.

## Full List of Methods

We are only listing the methods that natively solve fixed point problems.

### SpeedMapping.jl

  - `SpeedMappingJL()`: accelerates the convergence of a mapping to a fixed point by the
    Alternating cyclic extrapolation algorithm (ACX).

### FixedPointAcceleration.jl

  - `FixedPointAccelerationJL()`: accelerates the convergence of a mapping to a fixed point
    by the Anderson acceleration algorithm and a few other methods.

### NLsolve.jl

In our tests, we have found the anderson method implemented here to NOT be the most
robust.

  - `NLsolveJL(; method = :anderson)`: Anderson acceleration for fixed point problems.

### SIAMFANLEquations.jl

  - `SIAMFANLEquationsJL(; method = :anderson)`: Anderson acceleration for fixed point problems.
