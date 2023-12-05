# Release Notes

## Breaking Changes in NonlinearSolve.jl v3

1. `GeneralBroyden` and `GeneralKlement` have been renamed to `Broyden` and `Klement`
   respectively.
2. Compat for `SimpleNonlinearSolve` has been bumped to `v1`.
3. The old API to specify autodiff via `Val` and chunksize (that was deprecated in v2) has
   been removed.
