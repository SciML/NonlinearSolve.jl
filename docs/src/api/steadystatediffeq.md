# SteadyStateDiffEq.jl

This is a wrapper package for using ODE solvers from
[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) into the SciML interface.
Note that these solvers do not come by default, and thus one needs to install
the package before using these solvers:

```julia
using Pkg
Pkg.add("SteadyStateDiffEq")
using SteadyStateDiffEq
```

These methods can be used independently of the rest of NonlinearSolve.jl

## Solver API

```@docs
DynamicSS
```
