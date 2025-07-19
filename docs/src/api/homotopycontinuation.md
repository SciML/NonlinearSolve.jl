# HomotopyContinuation.jl

NonlinearSolve wraps the homotopy continuation algorithm implemented in
HomotopyContinuation.jl. This solver is not included by default and needs
to be installed separately:

```julia
import Pkg
Pkg.add("NonlinearSolveHomotopyContinuation")
import NonlinearSolveHomotopyContinuation
import NonlinearSolve as NLS
```

# Solver API

```@docs
NonlinearSolveHomotopyContinuation.HomotopyContinuationJL
SciMLBase.HomotopyNonlinearFunction
```
