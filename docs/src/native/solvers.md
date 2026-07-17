# NonlinearSolve.jl Solvers

These are the native solvers of NonlinearSolve.jl.

```@index
Pages = ["solvers.md"]
```

## General Keyword Arguments

Several Algorithms share the same specification for common keyword arguments. Those are
documented in this section to avoid repetition. Certain algorithms might have additional
considerations for these keyword arguments, which are documented in the algorithm's
documentation.

  - `linsolve`: the [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) solvers used
    for the linear solves within the Newton method. Defaults to `nothing`, which means it
    uses the LinearSolve.jl default algorithm choice. For more information on available
    algorithm choices, see the
    [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
  - `linesearch`: the line search algorithm to use. Defaults to
    [`NoLineSearch()`](@extref LineSearch.NoLineSearch), which means that no line search is
    performed.
  - `autodiff`: determines the backend used for the Jacobian. Note that this
    argument is ignored if an analytical Jacobian is passed, as that will be used instead.
    Defaults to `nothing` which means that a default is selected according to the problem
    specification! Valid choices are types from ADTypes.jl.
  - `vjp_autodiff`: similar to `autodiff`, but is used to compute Jacobian
    Vector Products. Ignored if the NonlinearFunction contains the `jvp` function.
  - `vjp_autodiff`: similar to `autodiff`, but is used to compute Vector
    Jacobian Products. Ignored if the NonlinearFunction contains the `vjp` function.
  - `concrete_jac`: whether to build a concrete Jacobian. If a Krylov-subspace method is
    used, then the Jacobian will not be constructed and instead direct Jacobian-Vector
    products `J*v` are computed using forward-mode automatic differentiation or finite
    differencing tricks (without ever constructing the Jacobian). However, if the Jacobian
    is still needed, for example for a preconditioner, `concrete_jac = true` can be passed
    in order to force the construction of the Jacobian.
  - `forcing`: Adaptive forcing term strategy for Newton-Krylov methods. When using an
    iterative linear solver (Krylov method), this controls how accurately the linear system
    is solved at each Newton iteration. Defaults to `nothing` (fixed tolerance). See
    [Forcing Term Strategies](@ref forcing_strategies) for available options.
  - `jacobian_reuse`: controls whether a Jacobian can be reused across accepted nonlinear
    iterations. `nothing` or `false` (the default) uses a fresh Jacobian after every accepted
    step. `true` selects [`JacobianReuse()`](@ref), or a configured `JacobianReuse` policy can
    be supplied directly. An unchanged concrete linear system also reuses its factorization.

## Nonlinear Solvers

```@docs
NewtonRaphson
DFSane
Broyden
Klement
LimitedMemoryBroyden
```

## Homotopy / Continuation Solvers

```@docs
HomotopySweep
ArcLengthContinuation
HomotopyPolyAlgorithm
```

## Nonlinear Least Squares Solvers

```@docs
GaussNewton
```

## Both Nonlinear & Nonlinear Least Squares Solvers

These solvers can be used for both nonlinear and nonlinear least squares problems.

```@docs
TrustRegion
LevenbergMarquardt
PseudoTransient
```

## Polyalgorithms

```@docs
NonlinearSolvePolyAlgorithm
FastShortcutNonlinearPolyalg
FastShortcutNLLSPolyalg
RobustMultiNewton
```

## Advanced Solvers

All of the previously mentioned solvers are wrappers around the following solvers. These
are meant for advanced users and allow building custom solvers.

```@docs
QuasiNewtonAlgorithm
GeneralizedFirstOrderAlgorithm
GeneralizedDFSane
```

## Jacobian Reuse

```@docs
JacobianReuse
```

Jacobian reuse is most useful when constructing or factorizing the Jacobian dominates the
cost of evaluating the residual. It changes exact Newton iteration into a modified-Newton
iteration, which can require more nonlinear steps, so it is opt-in. For example:

```julia
sol = solve(prob, NewtonRaphson(jacobian_reuse = JacobianReuse()))
```

The same policy works with `TrustRegion`, `GaussNewton`, `LevenbergMarquardt`, and
`PseudoTransient`. Damped and matrix-free systems retain their normal linear-solver update
behavior. Rejected trust-region steps reuse a fresh Jacobian because the nonlinear state did
not change; a rejected step based on stale Jacobian information requests a refresh.

The policy is local to one nonlinear cache lifecycle and is reset by `reinit!`. An outer
solver that owns a related but distinct operator should keep using the explicit
`step!(cache; recompute_jacobian = ...)` interface. In particular,
OrdinaryDiffEqNonlinearSolve distinguishes the ODE Jacobian `J` from
the iteration matrix `W` assembled from `J`, the mass matrix, `γ`, and `dt`; it decides
independently when each must be rebuilt and retains convergence information across time
steps. Its explicit decision takes precedence over this standalone policy.

## [Forcing Term Strategies](@id forcing_strategies)

Forcing term strategies control how accurately the linear system is solved at each Newton
iteration when using iterative (Krylov) linear solvers. This is the key idea behind
Newton-Krylov methods: instead of solving ``J \delta u = -f(u)`` exactly, we solve it only
approximately with a tolerance ``\eta_k`` (the forcing term).

The [eisenstat1996choosing](@citet) paper introduced adaptive strategies for choosing
``\eta_k`` that can significantly improve convergence, especially for problems where the
initial guess is far from the solution.

```@docs
EisenstatWalkerForcing2
```

### Example Usage

```julia
using NonlinearSolve, LinearSolve

# Define a large nonlinear problem
function f!(F, u, p)
    for i in 2:(length(u) - 1)
        F[i] = u[i - 1] - 2u[i] + u[i + 1] + sin(u[i])
    end
    F[1] = u[1] - 1.0
    F[end] = u[end]
end

n = 1000
u0 = zeros(n)
prob = NonlinearProblem(f!, u0)

# Use Newton-Raphson with GMRES and Eisenstat-Walker forcing
sol = solve(prob, NewtonRaphson(
    linsolve = KrylovJL_GMRES(),
    forcing = EisenstatWalkerForcing2()
))
```
