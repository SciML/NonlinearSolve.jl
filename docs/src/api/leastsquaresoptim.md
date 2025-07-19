# LeastSquaresOptim.jl

This is an extension for importing solvers from LeastSquaresOptim.jl into the SciML
interface. Note that these solvers do not come by default, and thus one needs to install
the package before using these solvers:

```julia
import Pkg
Pkg.add("LeastSquaresOptim")
import LeastSquaresOptim
import NonlinearSolve as NLS
```

## Solver API

```@docs
LeastSquaresOptimJL
```
