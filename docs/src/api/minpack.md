# MINPACK.jl

This is a extension for importing solvers from MINPACK into the SciML interface. Note that
these solvers do not come by default, and thus one needs to install the package before using
these solvers:

```julia
using Pkg
Pkg.add("MINPACK")
using MINPACK, NonlinearSolve
```

## Solver API

```@docs
CMINPACK
```
