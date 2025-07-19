# FastLevenbergMarquardt.jl

This is an extension for importing solvers from FastLevenbergMarquardt.jl into the SciML
interface. Note that these solvers do not come by default, and thus one needs to install
the package before using these solvers:

```julia
import Pkg
Pkg.add("FastLevenbergMarquardt")
import FastLevenbergMarquardt
import NonlinearSolve as NLS
```

## Solver API

```@docs
FastLevenbergMarquardtJL
```
