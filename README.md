# NonlinearSolve.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/NonlinearSolve/stable/)

[![codecov](https://codecov.io/gh/SciML/NonlinearSolve.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/NonlinearSolve.jl)
[![Build Status](https://github.com/SciML/NonlinearSolve.jl/workflows/CI/badge.svg)](https://github.com/SciML/NonlinearSolve.jl/actions?query=workflow%3ACI)
[![Build status](https://badge.buildkite.com/413dc8df7d555cc14c262aba066503a9e7a42023f9cfb75a55.svg)](https://buildkite.com/julialang/nonlinearsolve-dot-jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Fast implementations of root finding algorithms in Julia that satisfy the SciML common interface.

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/NonlinearSolve/stable/). Use the
[in-development documentation](https://docs.sciml.ai/NonlinearSolve/dev/) for the version of
the documentation which contains the unreleased features.

## High Level Examples

```julia
using NonlinearSolve, StaticArrays

f(u, p) = u .* u .- 2
u0 = @SVector[1.0, 1.0]
probN = NonlinearProblem(f, u0)
solver = solve(probN, NewtonRaphson(), abstol = 1e-9)

## Bracketing Methods

f(u, p) = u .* u .- 2.0
u0 = (1.0, 2.0) # brackets
probB = IntervalNonlinearProblem(f, u0)
sol = solve(probB, ITP())
```

## v1.0 Breaking Release Highlights!

v1.0 has been released for NonlinearSolve.jl, making it a decentralized solver library
akin to DifferentialEquations.jl. For simple implementations of nonlinear solvers,
you can now use SimpleNonlinearSolve.jl. `Falsi`, `Bisection`, and `NewtonRaphson`
implementations designed for scalar and static vector inputs have all moved to the
lower dependency version. NonlinearSolve.jl is thus designed for the larger scale
more complex implementations, with `NewtonRaphson` now sporting support for
LinearSolve.jl and soon SparseDiffTools.jl to allow for preconditioned Newton-Krylov and
exploitation of sparsity. The two pieces will continue to grow in this direction,
with NonlinearSolve.jl gaining more and more wrapped solver libraries and support
for more complex methods, while SimpleNonlinearSolve.jl will keep a lower dependency
version with implementations for small scale problems that do not need all of the
extra tooling.

Additionally, `NonlinearProblem` was split into `NonlinearProblem` and `IntervalNonlinearProblem`,
i.e. the bracketing versions now have their own problem definition, rather than using
a `Tuple` for `u0` in a `NonlinearProblem`. This helps for finding problem-algorithm
pairing errors at type time and overall improves the documentation / makes the roles
more clear.
