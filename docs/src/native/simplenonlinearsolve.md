# SimpleNonlinearSolve.jl

These methods can be used independently of the rest of NonlinearSolve.jl

```@index
Pages = ["simplenonlinearsolve.md"]
```

## Interval Methods

These methods are suited for interval (scalar) root-finding problems,
i.e. `IntervalNonlinearProblem`.

```@docs
ITP
Alefeld
Bisection
Falsi
Ridder
Brent
```

## General Methods

These methods are suited for any general nonlinear root-finding problem, i.e.
`NonlinearProblem`.

| Solver                               | In-place | Out of Place | Non-Allocating (Scalars) | Non-Allocating (`SArray`) |
| ------------------------------------ | -------- | ------------ | ------------------------ | ------------------------- |
| [`SimpleNewtonRaphson`](@ref)        | ✔️        | ✔️            | ✔️                        | ✔️                         |
| [`SimpleBroyden`](@ref)              | ✔️        | ✔️            | ✔️                        | ✔️                         |
| [`SimpleHalley`](@ref)               | ❌        | ✔️            | ✔️                        | ❌                         |
| [`SimpleKlement`](@ref)              | ✔️        | ✔️            | ✔️                        | ✔️                         |
| [`SimpleTrustRegion`](@ref)          | ✔️        | ✔️            | ✔️                        | ✔️                         |
| [`SimpleDFSane`](@ref)               | ✔️        | ✔️            | ✔️[^1]                    | ✔️                         |
| [`SimpleLimitedMemoryBroyden`](@ref) | ✔️        | ✔️            | ✔️                        | ✔️[^2]                     |

The algorithms which are non-allocating can be used directly inside GPU Kernels[^3].
See [PSOGPU.jl](https://github.com/SciML/PSOGPU.jl) for more details.

[^1]: Needs [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl) to be
      installed and loaded for the non-allocating version.
[^2]: This method is non-allocating if the termination condition is set to either `nothing`
      (default) or [`AbsNormTerminationMode`](@ref).
[^3]: Only the defaults are guaranteed to work inside kernels. We try to provide warnings
      if the used version is not non-allocating.

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
