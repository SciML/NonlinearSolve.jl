# SimpleNonlinearSolve.jl

These methods can be used independently of the rest of NonlinearSolve.jl

```@index
Pages = ["simplenonlinearsolve.md"]
```

## General Methods

These methods are suited for any general nonlinear root-finding problem, i.e.
`NonlinearProblem`.

| Solver                               | In-place | Out of Place | Non-Allocating (Scalars) | Non-Allocating (`SArray`) |
|:------------------------------------ |:-------- |:------------ |:------------------------ |:------------------------- |
| [`SimpleNewtonRaphson`](@ref)        | ✔️       | ✔️           | ✔️                       | ✔️                        |
| [`SimpleBroyden`](@ref)              | ✔️       | ✔️           | ✔️                       | ✔️                        |
| [`SimpleHalley`](@ref)               | ❌        | ✔️           | ✔️                       | ❌                         |
| [`SimpleKlement`](@ref)              | ✔️       | ✔️           | ✔️                       | ✔️                        |
| [`SimpleTrustRegion`](@ref)          | ✔️       | ✔️           | ✔️                       | ✔️                        |
| [`SimpleDFSane`](@ref)               | ✔️       | ✔️           | ✔️[^1]                   | ✔️                        |
| [`SimpleLimitedMemoryBroyden`](@ref) | ✔️       | ✔️           | ✔️                       | ✔️[^2]                    |

The algorithms which are non-allocating can be used directly inside GPU Kernels[^3].
See [ParallelParticleSwarms.jl](https://github.com/SciML/ParallelParticleSwarms.jl) for more details.

```@docs
SimpleNewtonRaphson
SimpleBroyden
SimpleHalley
SimpleKlement
SimpleTrustRegion
SimpleDFSane
SimpleLimitedMemoryBroyden
```

`SimpleGaussNewton` is aliased to [`SimpleNewtonRaphson`](@ref) for solving Nonlinear Least
Squares problems.

[^1]: Needs [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl) to be
    installed and loaded for the non-allocating version.
[^2]: This method is non-allocating if the termination condition is set to either `nothing`
    (default) or [`NonlinearSolveBase.AbsNormTerminationMode`](@ref).
[^3]: Only the defaults are guaranteed to work inside kernels. We try to provide warnings
    if the used version is not non-allocating.
