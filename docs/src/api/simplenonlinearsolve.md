# SimpleNonlinearSolve.jl

These methods can be used independently of the rest of NonlinearSolve.jl

## Solver API

### Interval Methods

These methods are suited for interval (scalar) root-finding problems, i.e. `IntervalNonlinearProblem`.

```@docs
ITP
Bisection
Falsi
Ridder
Brent
```

### General Methods

These methods are suited for any general nonlinear root-finding problem, i.e. `NonlinearProblem`.

```@docs
SimpleNewtonRaphson
Broyden
SimpleHalley
Klement
SimpleTrustRegion
SimpleDFSane
LBroyden
```
