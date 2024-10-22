# SimpleNonlinearSolve.jl

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
```
