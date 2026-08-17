# [Termination Conditions](@id termination_condition)

Provides a API to specify termination conditions for [`NonlinearProblem`](@ref) and
[`SteadyStateProblem`](@ref). For details on the various termination modes:

## Termination Conditions

The termination condition is constructed as:

```julia
cache = init(du, u, AbsNormSafeBestTerminationMode(); abstol = 1e-9, reltol = 1e-9)
```

If `abstol` and `reltol` are not supplied, then we choose a default based on the element
types of `du` and `u`.

!!! note

    The first argument (written `du` here and ``\Delta u`` in the mode docstrings) is the
    **residual** `f(u)`, not the Newton increment — every solver in this package passes the
    residual. The naming is inherited from the step-based use of the same machinery in
    DifferentialEquations.jl.

    A consequence worth knowing: the default termination mode for a `NonlinearProblem` is an
    *absolute* one, so `reltol` has no effect on it. Solving the same problem with
    `reltol = 1e-1` and `reltol = 1e-8` gives an identical iteration count, f-evaluation count
    and residual; only `abstol` moves them.

To test for termination simply call the `cache`:

```julia
terminated = cache(du, u, uprev)
```

### Absolute Tolerance

```@docs
NonlinearSolveBase.AbsTerminationMode
NonlinearSolveBase.AbsNormTerminationMode
NonlinearSolveBase.AbsNormSafeTerminationMode
NonlinearSolveBase.AbsNormSafeBestTerminationMode
```

### Relative Tolerance

```@docs
NonlinearSolveBase.RelTerminationMode
NonlinearSolveBase.RelNormTerminationMode
NonlinearSolveBase.RelNormSafeTerminationMode
NonlinearSolveBase.RelNormSafeBestTerminationMode
```

### Both Tolerances

```@docs
NonlinearSolveBase.NormTerminationMode
```
