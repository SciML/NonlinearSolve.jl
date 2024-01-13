# SimpleNonlinearSolve.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/NonlinearSolve/stable/)

[![codecov](https://codecov.io/gh/SciML/SimpleNonlinearSolve.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SciML/SimpleNonlinearSolve.jl)
[![Build Status](https://github.com/SciML/SimpleNonlinearSolve.jl/workflows/CI/badge.svg)](https://github.com/SciML/SimpleNonlinearSolve.jl/actions?query=workflow%3ACI)
[![Build status](https://badge.buildkite.com/c5f7db4f1b5e8a592514378b6fc807d934546cc7d5aa79d645.svg?branch=main)](https://buildkite.com/julialang/simplenonlinearsolve-dot-jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Fast implementations of root finding algorithms in Julia that satisfy the SciML common interface.
SimpleNonlinearSolve.jl focuses on low-dependency implementations of very fast methods for
very small and simple problems. For the full set of solvers, see
[NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl), of which
SimpleNonlinearSolve.jl is just one solver set.

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/NonlinearSolve/stable/). Use the
[in-development documentation](https://docs.sciml.ai/NonlinearSolve/dev/) for the version of
the documentation which contains the unreleased features.

## High Level Examples

```julia
using SimpleNonlinearSolve, StaticArrays

f(u, p) = u .* u .- 2
u0 = @SVector[1.0, 1.0]
probN = NonlinearProblem{false}(f, u0)
solver = solve(probN, SimpleNewtonRaphson(), abstol = 1e-9)

## Bracketing Methods

f(u, p) = u .* u .- 2.0
u0 = (1.0, 2.0) # brackets
probB = IntervalNonlinearProblem(f, u0)
sol = solve(probB, ITP())
```

For more details on the bracketing methods, refer to the [Tutorials](https://docs.sciml.ai/NonlinearSolve/stable/tutorials/nonlinear/#Using-Bracketing-Methods) and detailed [APIs](https://docs.sciml.ai/NonlinearSolve/stable/api/simplenonlinearsolve/#Solver-API)

## Breaking Changes in v1.0.0

  - Batched solvers have been removed in favor of `BatchedArrays.jl`. Stay tuned for detailed
    tutorials on how to use `BatchedArrays.jl` with `NonlinearSolve` & `SimpleNonlinearSolve`
    solvers.
  - The old style of specifying autodiff with `chunksize`, `standardtag`, etc. has been
    deprecated in favor of directly specifying the autodiff type, like `AutoForwardDiff`.
  - `Broyden` and `Klement` have been renamed to `SimpleBroyden` and `SimpleKlement` to
    avoid conflicts with `NonlinearSolve.jl`'s `GeneralBroyden` and `GeneralKlement`, which
    will be renamed to `Broyden` and `Klement` in the future.
  - `LBroyden` has been renamed to `SimpleLimitedMemoryBroyden` to make it consistent with
    `NonlinearSolve.jl`'s `LimitedMemoryBroyden`.
