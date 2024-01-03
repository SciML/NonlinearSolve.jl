# NonlinearSolve.jl Native Solvers

These are the native solvers of NonlinearSolve.jl.

## General Keyword Arguments

Several Algorithms share the same specification for common keyword arguments. Those are
documented in this section to avoid repetition. Certain algorithms might have additional
considerations for these keyword arguments, which are documented in the algorithm's
documentation.

  * `linsolve`: the [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) solvers used
    for the linear solves within the Newton method. Defaults to `nothing`, which means it
    uses the LinearSolve.jl default algorithm choice. For more information on available
    algorithm choices, see the
    [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
  * `precs`: the choice of preconditioners for the linear solver. Defaults to using no
    preconditioners. For more information on specifying preconditioners for LinearSolve
    algorithms, consult the
    [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
  * `linesearch`: the line search algorithm to use. Defaults to [`NoLineSearch()`](@ref),
    which means that no line search is performed.  Algorithms from `LineSearches.jl` must be
    wrapped in `LineSearchesJL` before being supplied.
  * `autodiff`/`jacobian_ad`: etermines the backend used for the Jacobian. Note that this
    argument is ignored if an analytical Jacobian is passed, as that will be used instead.
    Defaults to `nothing` which means that a default is selected according to the problem
    specification! Valid choices are types from ADTypes.jl.
  * `forward_ad`/`vjp_autodiff`: similar to `autodiff`, but is used to compute Jacobian
    Vector Products. Ignored if the NonlinearFunction contains the `jvp` function.
  * `reverse_ad`/`vjp_autodiff`: similar to `autodiff`, but is used to compute Vector
    Jacobian Products. Ignored if the NonlinearFunction contains the `vjp` function.
  * `concrete_jac`: whether to build a concrete Jacobian. If a Krylov-subspace method is
    used, then the Jacobian will not be constructed and instead direct Jacobian-Vector
    products `J*v` are computed using forward-mode automatic differentiation or finite
    differencing tricks (without ever constructing the Jacobian). However, if the Jacobian
    is still needed, for example for a preconditioner, `concrete_jac = true` can be passed
    in order to force the construction of the Jacobian.

## Nonlinear Solvers

```@docs
NewtonRaphson
PseudoTransient
DFSane
Broyden
Klement
LimitedMemoryBroyden
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
```

## Polyalgorithms

```@docs
NonlinearSolvePolyAlgorithm
FastShortcutNonlinearPolyalg
FastShortcutNLLSPolyalg
RobustMultiNewton
```

## Radius Update Schemes for Trust Region (RadiusUpdateSchemes)

```@docs
RadiusUpdateSchemes
```

### Available Radius Update Schemes

```@docs
RadiusUpdateSchemes.Simple
RadiusUpdateSchemes.Hei
RadiusUpdateSchemes.Yuan
RadiusUpdateSchemes.Bastin
RadiusUpdateSchemes.Fan
RadiusUpdateSchemes.NLsolve
RadiusUpdateSchemes.NocedalWright
```
