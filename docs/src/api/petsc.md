# PETSc.jl

This is a extension for importing solvers from PETSc.jl SNES into the SciML interface. Note
that these solvers do not come by default, and thus one needs to install the package before
using these solvers:

```julia
import Pkg
Pkg.add("PETSc")
import PETSc
import NonlinearSolve as NLS
```

## Solver API

```@docs
PETScSNES
```
