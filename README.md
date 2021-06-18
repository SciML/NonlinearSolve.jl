# NonlinearSolve.jl

[![Github Action CI](https://github.com/SciML/NonlinearSolve.jl/workflows/CI/badge.svg)](https://github.com/SciML/NonlinearSolve.jl/actions)
[![Coverage Status](https://coveralls.io/repos/github/SciML/NonlinearSolve.jl/badge.svg?branch=master)](https://coveralls.io/github/SciML/NonlinearSolve.jl?branch=master)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](http://nlsolve.sciml.ai/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](http://nlsolve.sciml.ai/dev/)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

Fast implementations of root finding algorithms in Julia that satisfy the SciML common interface.

For information on using the package,
[see the stable documentation](https://mtk.sciml.ai/stable/). Use the
[in-development documentation](https://mtk.sciml.ai/dev/) for the version of
the documentation which contains the unreleased features.

## High Level Examples

```julia
using NonlinearSolve, StaticArrays

f(u,p) = u .* u .- 2
u0 = @SVector[1.0, 1.0]
probN = NonlinearProblem{false}(f, u0)
solver = solve(probN, NewtonRaphson(), tol = 1e-9)

## Bracketing Methods

f(u, p) = u .* u .- 2.0
u0 = (1.0, 2.0) # brackets
probB = NonlinearProblem(f, u0)
sol = solve(probB, Falsi())
```
