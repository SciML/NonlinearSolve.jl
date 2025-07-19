# Sundials.jl

This is a wrapper package for importing solvers from Sundials into the SciML interface.
Note that these solvers do not come by default, and thus one needs to install the package
before using these solvers:

```julia
import Pkg
Pkg.add("Sundials")
import Sundials
```

These methods can be used independently of the rest of NonlinearSolve.jl.

## Solver API

```@docs
KINSOL
```
