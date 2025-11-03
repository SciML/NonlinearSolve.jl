# [Common Solver Options (Solve Keyword Arguments)](@id solver_options)

```@docs
solve(::NonlinearProblem, args...; kwargs...)
```

## General Controls

  - `alias_u0::Bool`: Whether to alias the initial condition or use a copy.
    Defaults to `false`.
  - `internalnorm::Function`: The norm used by the solver. Default depends on algorithm
    choice.

## Iteration Controls

  - `maxiters::Int`: The maximum number of iterations to perform. Defaults to `1000`.
  - `maxtime`: The maximum time for solving the nonlinear system of equations. Defaults to
    `nothing` which means no time limit. Note that setting a time limit does have a small
    overhead.
  - `abstol::Number`: The absolute tolerance. Defaults to
    `real(oneunit(T)) * (eps(real(one(T))))^(4 // 5)`.
  - `reltol::Number`: The relative tolerance. Defaults to
    `real(oneunit(T)) * (eps(real(one(T))))^(4 // 5)`.
  - `termination_condition`: Termination Condition from NonlinearSolveBase. Defaults to
    `AbsSafeBestTerminationMode()` for `NonlinearSolve.jl` and `AbsTerminateMode()` for
    `SimpleNonlinearSolve.jl`.

## Tracing Controls

These are exclusively available for native `NonlinearSolve.jl` solvers.

  - `show_trace`: Must be `Val(true)` or `Val(false)`. This controls whether the trace is
    displayed to the console. (Defaults to `Val(false)`)
  - `trace_level`: Needs to be one of Trace Objects: [`TraceMinimal`](@ref),
    [`TraceWithJacobianConditionNumber`](@ref), or [`TraceAll`](@ref). This controls the
    level of detail of the trace. (Defaults to `TraceMinimal()`)
  - `store_trace`: Must be `Val(true)` or `Val(false)`. This controls whether the trace is
    stored in the solution object. (Defaults to `Val(false)`)

## Verbosity Controls

NonlinearSolve.jl provides fine-grained control over diagnostic messages, warnings, and errors
through the `verbose` keyword argument. The verbosity system allows you to control what
information is displayed during the solve process. See [SciMLLogging.jl](https://docs.sciml.ai/SciMLLogging/dev/) for more details. 

```@docs
NonlinearVerbosity
```

### Quick Start

```julia
# Use a preset
solve(prob, alg; verbose = NonlinearVerbosity(SciMLLogging.Standard()))

# Silence all messages
solve(prob, alg; verbose = NonlinearVerbosity(SciMLLogging.None()))

# Maximum verbosity
solve(prob, alg; verbose = NonlinearVerbosity(SciMLLogging.All()))

# Custom configuration
solve(prob, alg; verbose = NonlinearVerbosity(
    alias_u0_immutable = SciMLLogging.WarnLevel(),
    threshold_state = SciMLLogging.InfoLevel()
))
```