# Multi-Level Nonlinear Solvers

Solvers for systems whose unknowns split into a globally coupled block `ū` and an internal
block `q` that decouples into one small independent problem per point. See
[Multi-Level Newton for Problems with Internal Variables](@ref multilevel_newton) for the
contract these callbacks are bound by and a worked example.

```@index
Pages = ["multilevelnonlinearsolve.md"]
```

## Solver

```@docs
MultiLevelNewton
```

## Problem Specification

```@docs
MultiLevelNonlinearFunction
```

## Local Forcing

How accurately the internal variables are eliminated at each global iteration. This is the
nonlinear, local counterpart of the [linear forcing](@ref forcing_strategies) applied to the
condensed linear solve; the two are independent.

```@docs
LocalToleranceSchedule
LocalForcingParameters
local_tolerance
user_parameters
```

## Local Ensemble Helpers

Chunk-parallel execution of the per-point problems, and the double buffer that keeps trial
evaluations off the committed internal state.

```@docs
LocalEnsemble
ensemble_foreach
LocalStateBuffer
committed_state
trial_state
commit_local_state!
```

## Full-Space Arm

The alternative arrangement: the solver iterates on the whole `[ū; q]`, a δq-zeroing linear
solver keeps the step inside the primary block, and the commit runs as a `postcondition`
corrector. Plain Newton only — see the tutorial for why globalization is out of scope here.

```@docs
fullspace_problem
SchurOperator
CondensedFactorization
MultiLevelProjection
```

## Diagnostics

```@docs
MultiLevelNonlinearSolve.ncommits
```
