# NonlinearSolve.jl: High-Performance Unified Nonlinear Solvers

NonlinearSolve.jl is a unified interface for the nonlinear solving packages of
Julia. It includes its own high-performance nonlinear solvers which include the
ability to swap out to fast direct and iterative linear solvers, along with the
ability to use sparse automatic differentiation for Jacobian construction and
Jacobian-vector products. It interfaces with other packages of the Julia ecosystem
to make it easy to test alternative solver packages and pass small types to
control algorithm swapping. It also interfaces with the
[ModelingToolkit.jl](https://mtk.sciml.ai/dev/) world of symbolic modeling to
allow for automatically generating high-performance code.

Performance is key: the current methods are made to be highly performant on
scalar and statically sized small problems, with options for large-scale systems.
If you run into any performance issues, please file an issue. Note that this
package is distinct from [SciMLNLSolve.jl](https://github.com/SciML/SciMLNLSolve.jl).
Consult the [NonlinearSystemSolvers](@ref nonlinearsystemsolvers) page for
information on how to import solvers from different packages.

## Installation

To install NonlinearSolve.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("NonlinearSolve")
```

## Contributing

- Please refer to the
  [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
  for guidance on PRs, issues, and other matters relating to contributing to ModelingToolkit.
- There are a few community forums:
    - the #diffeq-bridged channel in the [Julia Slack](https://julialang.org/slack/)
    - [JuliaDiffEq](https://gitter.im/JuliaDiffEq/Lobby) on Gitter
    - on the Julia Discourse forums (look for the [modelingtoolkit tag](https://discourse.julialang.org/tag/modelingtoolkit)
    - see also [SciML Community page](https://sciml.ai/community/)

## Roadmap

The current algorithms should support automatic differentiation, though improved
adjoint overloads are planned to be added in the current update (which will make
use of the `f(u,p)` form). Future updates will include standard methods for
larger scale nonlinear solving like Newton-Krylov methods.
