# LeastSquaresOptim.jl

This is a wrapper package for importing solvers from LeastSquaresOptim.jl into the SciML interface.
Note that these solvers do not come by default, and thus one needs to install
the package before using these solvers:

```julia
using Pkg
Pkg.add("LeastSquaresOptim")
using LeastSquaresOptim
```

These methods can be used independently of the rest of NonlinearSolve.jl

## Solver API

```@docs
LeastSquaresOptimJL
```