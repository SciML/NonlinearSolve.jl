# SpeedMapping.jl

This is a extension for importing solvers from SpeedMapping.jl into the SciML
interface. Note that these solvers do not come by default, and thus one needs to install
the package before using these solvers:

```julia
using Pkg
Pkg.add("SpeedMapping")
using SpeedMapping, NonlinearSolve
```

## Solver API

```@docs
SpeedMappingJL
```
