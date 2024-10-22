# [Termination Conditions](@id termination_condition)

Provides a API to specify termination conditions for [`NonlinearProblem`](@ref) and
[`SteadyStateProblem`](@ref). For details on the various termination modes:

## Termination Conditions

The termination condition is constructed as:

```julia
cache = init(du, u, AbsSafeBestTerminationMode(); abstol = 1e-9, reltol = 1e-9)
```

If `abstol` and `reltol` are not supplied, then we choose a default based on the element
types of `du` and `u`.

To test for termination simply call the `cache`:

```julia
terminated = cache(du, u, uprev)
```

### Absolute Tolerance

```@docs
AbsTerminationMode
AbsNormTerminationMode
AbsNormSafeTerminationMode
AbsNormSafeBestTerminationMode
```

### Relative Tolerance

```@docs
RelTerminationMode
RelNormTerminationMode
RelNormSafeTerminationMode
RelNormSafeBestTerminationMode
```
