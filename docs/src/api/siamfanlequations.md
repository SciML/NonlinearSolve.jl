# SIAMFANLEquations.jl

This is an extension for importing solvers from
[SIAMFANLEquations.jl](https://github.com/ctkelley/SIAMFANLEquations.jl) into the SciML
interface. Note that these solvers do not come by default, and thus one needs to install
the package before using these solvers:

```julia
import Pkg
Pkg.add("SIAMFANLEquations")
import SIAMFANLEquations
import NonlinearSolve as NLS
```

## Solver API

```@docs
SIAMFANLEquationsJL
```
