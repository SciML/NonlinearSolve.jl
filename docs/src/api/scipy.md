# SciPy

This is an extension for importing solvers from
[SciPy](https://scipy.org/) into the SciML
interface. Note that these solvers do not come by default, and thus one needs to install
the package before using these solvers:

```julia
import Pkg
Pkg.add("NonlinearSolveSciPy")
import NonlinearSolveSciPy as NLSP
```

Note that using this package requires Python and SciPy to be available via PythonCall.jl.

## Solver API

```@docs
NonlinearSolveSciPy.SciPyLeastSquares
NonlinearSolveSciPy.SciPyRoot
NonlinearSolveSciPy.SciPyRootScalar
```
