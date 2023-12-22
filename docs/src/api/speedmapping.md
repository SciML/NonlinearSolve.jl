# SppedMapping.jl

This is a extension for importing solvers from SppedMapping.jl into the SciML
interface. Note that these solvers do not come by default, and thus one needs to install
the package before using these solvers:

```julia
using Pkg
Pkg.add("SppedMapping")
using SppedMapping, NonlinearSolve
```

## Solver API

```@docs
SppedMappingJL
```
