# Optimization.jl

This is a extension for importing solvers from Optimization.jl into the SciML Nonlinear
Problem interface. Note that these solvers do not come by default, and thus one needs to
install the package before using these solvers:

```julia
using Pkg
Pkg.add("Optimization")
using Optimization, NonlinearSolve
```

## Solver API

```@docs
OptimizationJL
```
