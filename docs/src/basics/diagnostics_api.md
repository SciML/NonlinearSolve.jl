# [Diagnostics API](@id diagnostics_api)

Detailed API Documentation is provided at
[Diagnostics API Reference](@ref diagnostics_api_reference).

## Logging the Solve Process

All NonlinearSolve.jl native solvers allow storing and displaying the trace of the nonlinear
solve process. This is controlled by 3 keyword arguments to `solve`:

 1. `show_trace`: Must be `Val(true)` or `Val(false)`. This controls whether the trace is
    displayed to the console. (Defaults to `Val(false)`)
 2. `trace_level`: Needs to be one of Trace Objects: [`TraceMinimal`](@ref),
    [`TraceWithJacobianConditionNumber`](@ref), or [`TraceAll`](@ref). This controls the
    level of detail of the trace. (Defaults to `TraceMinimal()`)
 3. `store_trace`: Must be `Val(true)` or `Val(false)`. This controls whether the trace is
    stored in the solution object. (Defaults to `Val(false)`)

## Detailed Internal Timings

All the native NonlinearSolve.jl algorithms come with in-built
[TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl) support. However, this
is disabled by default and can be enabled via [`NonlinearSolveBase.enable_timer_outputs`](@ref).

Note that you will have to restart Julia to disable the timer outputs once enabled.

## Example Usage

```@example diagnostics_example
import NonlinearSolve as NLS

function nlfunc(resid, u0, p)
    resid[1] = u0[1] * u0[1] - p
    resid[2] = u0[2] * u0[2] - p
    resid[3] = u0[3] * u0[3] - p
    nothing
end

prob = NLS.NonlinearProblem(nlfunc, [1.0, 3.0, 5.0], 2.0)

NLS.solve(prob)
```

This produced the output, but it is hard to diagnose what is going on. We can turn on
the trace to see what is happening:

```@example diagnostics_example
NLS.solve(prob; show_trace = Val(true), trace_level = NLS.TraceAll(10))
nothing; # hide
```

You can also store the trace in the solution object:

```@example diagnostics_example
sol = NLS.solve(prob; trace_level = NLS.TraceAll(), store_trace = Val(true));

sol.trace
```

Now, let's try to investigate the time it took for individual internal steps. We will have
to use the `init` and `solve!` API for this. The `TimerOutput` will be present in
`cache.timer`. However, note that for poly-algorithms this is currently not implemented.

```@example diagnostics_example
cache = NLS.init(prob, NLS.NewtonRaphson(); show_trace = Val(true));
NLS.solve!(cache)
cache.timer
```

Let's try for some other solver:

```@example diagnostics_example
cache = NLS.init(prob, NLS.DFSane(); show_trace = Val(true), trace_level = NLS.TraceMinimal(50));
NLS.solve!(cache)
cache.timer
```

!!! note
    
    For `iteration == 0` only the `norm(fu, Inf)` is guaranteed to be meaningful. The other
    values being meaningful are solver dependent.
