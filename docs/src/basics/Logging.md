# Logging the Solve Process

All NonlinearSolve.jl native solvers allow storing and displaying the trace of the nonlinear
solve process. This is controlled by 3 keyword arguments to `solve`:

 1. `show_trace`: Must be `Val(true)` or `Val(false)`. This controls whether the trace is
    displayed to the console. (Defaults to `Val(false)`)
 2. `trace_level`: Needs to be one of Trace Objects: [`TraceMinimal`](@ref),
    [`TraceWithJacobianConditionNumber`](@ref), or [`TraceAll`](@ref). This controls the
    level of detail of the trace. (Defaults to `TraceMinimal()`)
 3. `store_trace`: Must be `Val(true)` or `Val(false)`. This controls whether the trace is
    stored in the solution object. (Defaults to `Val(false)`)

## Example Usage

```@example tracing
using ModelingToolkit, NonlinearSolve

@variables x y z
@parameters σ ρ β

# Define a nonlinear system
eqs = [0 ~ σ * (y - x),
    0 ~ x * (ρ - z) - y,
    0 ~ x * y - β * z]
@named ns = NonlinearSystem(eqs, [x, y, z], [σ, ρ, β])

u0 = [x => 1.0, y => 0.0, z => 0.0]

ps = [σ => 10.0 ρ => 26.0 β => 8 / 3]

prob = NonlinearProblem(ns, u0, ps)

solve(prob)
```

This produced the output, but it is hard to diagnose what is going on. We can turn on
the trace to see what is happening:

```@example tracing
solve(prob; show_trace = Val(true), trace_level = TraceAll(10))
```

You can also store the trace in the solution object:

```@example tracing
sol = solve(prob; trace_level = TraceAll(), store_trace = Val(true));

sol.trace
```

!!! note
    
    For `iteration == 0` only the `norm(fu, Inf)` is guaranteed to be meaningful. The other
    values being meaningful are solver dependent.

## API

```@docs
TraceMinimal
TraceWithJacobianConditionNumber
TraceAll
```
