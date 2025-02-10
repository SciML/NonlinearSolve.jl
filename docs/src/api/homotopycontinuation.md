# HomotopyContinuation.jl

NonlinearSolve wraps the homotopy continuation algorithm implemented in
HomotopyContinuation.jl. This solver is not included by default and needs
to be installed separately:

```julia
using Pkg
Pkg.add("NonlinearSolveHomotopyContinuation")
using NonlinearSolveHomotopyContinuation, NonlinearSolve
```

# Solver API

```@docs
NonlinearSolveHomotopyContinuation.HomotopyContinuationJL
SciMLBase.HomotopyNonlinearFunction
```
