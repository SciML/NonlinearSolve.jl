# NLSolvers.jl

This is a extension for importing solvers from NLSolvers.jl into the SciML interface. Note
that these solvers do not come by default, and thus one needs to install the package before
using these solvers:

```julia
using Pkg
Pkg.add("NLSolvers")
using NLSolvers, NonlinearSolve
```

## Solver API

```@docs
NLSolversJL
```
