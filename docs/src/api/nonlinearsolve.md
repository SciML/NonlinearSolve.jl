# NonlinearSolve.jl Native Solvers

These are the native solvers of NonlinearSolve.jl.

## Nonlinear Solvers

```@docs
NewtonRaphson
PseudoTransient
DFSane
Broyden
Klement
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
```
