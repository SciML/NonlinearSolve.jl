# Release Notes

## Breaking Changes in NonlinearSolve.jl v3

 1. `GeneralBroyden` and `GeneralKlement` have been renamed to `Broyden` and `Klement`
    respectively.
 2. Compat for `SimpleNonlinearSolve` has been bumped to `v1`.
 3. The old style of specifying autodiff with `chunksize`, `standardtag`, etc. has been
    deprecated in favor of directly specifying the autodiff type, like `AutoForwardDiff`.

## Breaking Changes in SimpleNonlinearSolve.jl v1

  - Batched solvers have been removed in favor of `BatchedArrays.jl`. Stay tuned for detailed
    tutorials on how to use `BatchedArrays.jl` with `NonlinearSolve` & `SimpleNonlinearSolve`
    solvers.
  - The old style of specifying autodiff with `chunksize`, `standardtag`, etc. has been
    deprecated in favor of directly specifying the autodiff type, like `AutoForwardDiff`.
  - `Broyden` and `Klement` have been renamed to `SimpleBroyden` and `SimpleKlement` to
    avoid conflicts with `NonlinearSolve.jl`'s `GeneralBroyden` and `GeneralKlement`, which
    will be renamed to `Broyden` and `Klement` in the future.
  - `LBroyden` has been renamed to `SimpleLimitedMemoryBroyden` to make it consistent with
    `NonlinearSolve.jl`'s `LimitedMemoryBroyden`.
