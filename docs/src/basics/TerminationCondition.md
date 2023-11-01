# [Termination Conditions](@id termination_condition)

Provides a API to specify termination conditions for [`NonlinearProblem`](@ref) and
[`SteadyStateProblem`](@ref). For details on the various termination modes:

## Termination Conditions

The termination condition is constructed as:

```julia
cache = init(du, u, AbsNormTerminationMode(); abstol = 1e-9, reltol = 1e-9)
```

If `abstol` and `reltol` are not supplied, then we choose a default based on the element
types of `du` and `u`.

We can query the `cache` using `DiffEqBase.get_termination_mode`, `DiffEqBase.get_abstol`
and `DiffEqBase.get_reltol`.

To test for termination simply call the `cache`:

```julia
terminated = cache(du, u, uprev)
```

!!! note
    
    The default for NonlinearSolve.jl is `AbsSafeBestTerminationMode`!

### Absolute Tolerance

```@docs
AbsTerminationMode
AbsNormTerminationMode
AbsSafeTerminationMode
AbsSafeBestTerminationMode
```

### Relative Tolerance

```@docs
RelTerminationMode
RelNormTerminationMode
RelSafeTerminationMode
RelSafeBestTerminationMode
```

### Both Absolute and Relative Tolerance

```@docs
NormTerminationMode
SteadyStateDiffEqTerminationMode
SimpleNonlinearSolveTerminationMode
```

### Return Codes

```@docs
DiffEqBase.NonlinearSafeTerminationReturnCode
DiffEqBase.NonlinearSafeTerminationReturnCode.Success
DiffEqBase.NonlinearSafeTerminationReturnCode.Default
DiffEqBase.NonlinearSafeTerminationReturnCode.Failure
DiffEqBase.NonlinearSafeTerminationReturnCode.PatienceTermination
DiffEqBase.NonlinearSafeTerminationReturnCode.ProtectiveTermination
```

## [Deprecated] Termination Condition API

!!! warning
    
    This is deprecated. Currently only parts of `SimpleNonlinearSolve` uses this API. That
    will also be phased out soon!

```@docs
NLSolveTerminationCondition
```
