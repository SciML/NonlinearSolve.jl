# NonlinearSolve.jl Native Solvers

These are the native solvers of NonlinearSolve.jl.

## Core Nonlinear Solvers

```@docs
NewtonRaphson
TrustRegion
PseudoTransient
DFSane
GeneralBroyden
GeneralKlement
```

## Polyalgorithms

```@docs
NonlinearSolvePolyAlgorithm
FastShortcutNonlinearPolyalg
FastShortcutNLLSPolyalg
RobustMultiNewton
```

## Nonlinear Least Squares Solvers

```@docs
LevenbergMarquardt
GaussNewton
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
