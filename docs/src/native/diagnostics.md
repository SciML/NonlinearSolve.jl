# [Diagnostics API](@id diagnostics_api_reference)

## Timer Outputs

These functions are not exported since the names have a potential for conflict.

```@docs
NonlinearSolve.enable_timer_outputs
NonlinearSolve.disable_timer_outputs
NonlinearSolve.@static_timeit
```

## Tracing API

```@docs
TraceAll
TraceWithJacobianConditionNumber
TraceMinimal
```

For details about the arguments refer to the documentation of
[`NonlinearSolve.AbstractNonlinearSolveTraceLevel`](@ref).
