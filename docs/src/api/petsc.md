# PETSc.jl

This is a extension for importing solvers from PETSc.jl SNES into the SciML interface. Note
that these solvers do not come by default, and thus one needs to install the package before
using these solvers:

```julia
using Pkg
Pkg.add("PETSc")
using PETSc, NonlinearSolve
```

## Solver API

```@docs
PETScSNES
```
