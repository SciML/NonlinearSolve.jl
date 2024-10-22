# BracketingNonlinearSolve.jl

Fast implementations of interval root finding algorithms in Julia that satisfy the SciML
common interface. BracketingNonlinearSolve.jl focuses on low-dependency implementations of
very fast methods for very small and simple problems. For the full set of solvers, see
[NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl), of which
BracketingNonlinearSolve.jl is just one solver set.

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/NonlinearSolve/stable/). Use the
[in-development documentation](https://docs.sciml.ai/NonlinearSolve/dev/) for the version of
the documentation which contains the unreleased features.

## High Level Examples

```julia
using BracketingNonlinearSolve

f(u, p) = u .* u .- 2.0
u0 = (1.0, 2.0) # brackets
probB = IntervalNonlinearProblem(f, u0)
sol = solve(probB, ITP())
```
