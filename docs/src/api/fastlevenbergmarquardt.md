# FastLevenbergMarquardt.jl

This is an extension for importing solvers from FastLevenbergMarquardt.jl into the SciML
interface. Note that these solvers do not come by default, and thus one needs to install
the package before using these solvers:

```julia
using Pkg
Pkg.add("FastLevenbergMarquardt")
using FastLevenbergMarquardt, NonlinearSolve
```

## Solver API

```@docs
FastLevenbergMarquardtJL
```
