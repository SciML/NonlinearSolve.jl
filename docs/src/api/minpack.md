# MINPACK.jl

This is a extension for importing solvers from MINPACK into the SciML interface. Note that
these solvers do not come by default, and thus one needs to install the package before using
these solvers:

```julia
import Pkg
Pkg.add("MINPACK")
import MINPACK
import NonlinearSolve as NLS
```

## Solver API

```@docs
CMINPACK
```
