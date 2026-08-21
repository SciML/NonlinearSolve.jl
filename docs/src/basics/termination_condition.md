# [Termination Conditions](@id termination_condition)

Provides a API to specify termination conditions for
`NonlinearProblem` and `SteadyStateProblem`.
For details on the various
termination modes:

## Termination Conditions

The termination condition is constructed as:

```julia
prob = NonlinearProblem((u, p) -> [u[1]^2 - 2], [1.0])
mode = AbsNormSafeBestTerminationMode(Base.Fix1(maximum, abs))
cache = init(prob, mode, [1.0], [1.0]; abstol = 1.0e-9, reltol = 1.0e-9)
```

If `abstol` and `reltol` are not supplied, then we choose a default based on the element
types of `du` and `u`.

!!! note

    The first state argument (written `du` here and `r` in the mode docstrings) is the
    **residual** `f(u)`, not the Newton increment. The name `du` is inherited from the
    step-based use of the same machinery in DifferentialEquations.jl.

    A consequence worth knowing: the default termination mode for a `NonlinearProblem` is an
    *absolute* one, so `reltol` has no effect on it. To use a relative criterion, pass a
    `RelTerminationMode` or `RelNormTerminationMode` explicitly through
    `termination_condition`.

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
