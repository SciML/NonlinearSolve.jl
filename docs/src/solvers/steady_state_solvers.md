# [Steady State Solvers](@id ss_solvers)

```julia
solve(prob::SteadyStateProblem, alg; kwargs)
```

Solves for the steady states in the problem defined by `prob` using the algorithm `alg`. If
no algorithm is given, a default algorithm will be chosen.

## Recommended Methods

Conversion to a NonlinearProblem is generally the fastest method. However, this will not
guarantee the preferred root (the stable equilibrium), and thus if the preferred root is
required, then it's recommended that one uses [`DynamicSS`](@ref). For [`DynamicSS`](@ref),
often an adaptive stiff solver, like a Rosenbrock or BDF method (`Rodas5` or `QNDF`), is a
good way to allow for very large time steps as the steady state approaches.

The SteadyStateDiffEq.jl methods on a [`SteadyStateProblem`](@ref) respect the time
definition in the nonlinear definition, i.e., `u' = f(u, t)` uses the correct values for
`t` as the solution evolves. A conversion of a [`SteadyStateProblem`](@ref) to a
[`NonlinearProblem`](@ref) replaces this with the nonlinear system `u' = f(u, ∞)`, and thus
the direct [`SteadyStateProblem`](@ref) approach can give different answers (i.e., the
correct unique fixed point) on ODEs with non-autonomous dynamics.

If you have an unstable equilibrium and you want to solve for the unstable equilibrium,
then [`DynamicSS`](@ref) will not converge to that equilibrium for any initial condition.
However, Nonlinear Solvers don't suffer from this issue, and thus it's recommended to
use a nonlinear solver if you want to solve for the unstable equilibrium.

## Full List of Methods

### Conversion to NonlinearProblem

Any [`SteadyStateProblem`](@ref) can be trivially converted to a [`NonlinearProblem`](@ref)
via `NonlinearProblem(prob::SteadyStateProblem)`. Using this approach, any of the solvers
from the [Nonlinear System Solvers page](@ref nonlinearsystemsolvers) can be used. As a
convenience, users can use:

  - [`SSRootfind`](@ref): A wrapper around `NonlinearSolve.jl` compliant solvers which
    converts the [`SteadyStateProblem`](@ref) to a [`NonlinearProblem`](@ref) and solves it.

### SteadyStateDiffEq.jl

SteadyStateDiffEq.jl uses ODE solvers to iteratively approach the steady state. It is a
very stable method for solving nonlinear systems,
though often computationally more expensive than direct methods.

  - [`DynamicSS`](@ref) : Uses an ODE solver to find the steady state. Automatically
    terminates when close to the steady state. `DynamicSS(alg; tspan = Inf)` requires that
    an ODE algorithm is given as the first argument. The absolute and relative tolerances
    specify the termination conditions on the derivative's closeness to zero. This
    internally uses the `TerminateSteadyState` callback from the Callback Library. The
    simulated time, for which the ODE is solved, can be limited by `tspan`.  If `tspan` is a
    number, it is equivalent to passing `(zero(tspan), tspan)`.

Example usage:

```julia
import NonlinearSolve as NLS
import SteadyStateDiffEq as SSDE
import OrdinaryDiffEq as ODE
sol = NLS.solve(prob, SSDE.DynamicSS(ODE.Tsit5()))

import Sundials
sol = NLS.solve(prob, SSDE.DynamicSS(Sundials.CVODE_BDF()), dt = 1.0)
```

!!! note
    
    If you use `CVODE_BDF` you may need to give a starting `dt` via `dt=....`.
