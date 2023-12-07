# [Nonlinear Solutions](@id solution)

```@docs
SciMLBase.NonlinearSolution
```

## Return Code

  - `ReturnCode.Success` - The nonlinear solve succeeded.
  - `ReturnCode.ConvergenceFailure` - The nonlinear solve failed to converge due to stalling
    or some limit of the solver was exceeded. For example, too many shrinks for trust
    region methods, number of resets for Broyden, etc.
  - `ReturnCode.Unstable` - This corresponds to
    `NonlinearSafeTerminationReturnCode.ProtectiveTermination` and is caused if the step-size
    of the solver was too large or the objective value became non-finite.
  - `ReturnCode.MaxIters` - The maximum number of iterations was reached.
