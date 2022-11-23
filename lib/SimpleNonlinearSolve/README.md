# NonlinearSolve.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/NonlinearSolve/stable/)

[![codecov](https://codecov.io/gh/SciML/NonlinearSolve.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/NonlinearSolve.jl)
[![Build Status](https://github.com/SciML/NonlinearSolve.jl/workflows/CI/badge.svg)](https://github.com/SciML/NonlinearSolve.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Fast implementations of root finding algorithms in Julia that satisfy the SciML common interface.
SimpleNonlinearSolve.jl focuses on low-dependency implementations of very fast methods for
very small and simple problems. For the full set of solvers, see NonlinearSolve.jl, of which
SimpleNonlinearSolve.jl is just one solver set.

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/NonlinearSolve/stable/). Use the
[in-development documentation](https://docs.sciml.ai/NonlinearSolve/dev/) for the version of
the documentation which contains the unreleased features.

## High Level Examples

```julia
using SimpleNonlinearSolve, StaticArrays

f(u,p) = u .* u .- 2
u0 = @SVector[1.0, 1.0]
probN = NonlinearProblem{false}(f, u0)
solver = solve(probN, SimpleNewtonRaphson(), tol = 1e-9)

## Bracketing Methods

f(u, p) = u .* u .- 2.0
u0 = (1.0, 2.0) # brackets
probB = IntervalNonlinearProblem(f, u0)
sol = solve(probB, Falsi())
```
