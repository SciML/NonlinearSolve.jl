# SteadyStateDiffEq.jl

This is a wrapper package for using ODE solvers from
[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) into the SciML
interface. Note that these solvers do not come by default, and thus one needs to install
the package before using these solvers:

```julia
import Pkg
Pkg.add("SteadyStateDiffEq")
import SteadyStateDiffEq as SSDE
```

These methods can be used independently of the rest of NonlinearSolve.jl

```@index
Pages = ["steadystatediffeq.md"]
```

## Solver API

```@docs
DynamicSS
SSRootfind
```
