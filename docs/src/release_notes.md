# Release Notes

## Oct '24

### Breaking Changes in `NonlinearSolve.jl` v4

  - `ApproximateJacobianSolveAlgorithm` has been renamed to `QuasiNewtonAlgorithm`.
  - Preconditioners for the linear solver needs to be specified with the linear solver
    instead of `precs` keyword argument.
  - See [common breaking changes](@ref common-breaking-changes-v4v2) below.

### Breaking Changes in `SimpleNonlinearSolve.jl` v2

  - See [common breaking changes](@ref common-breaking-changes-v4v2) below.

### [Common Breaking Changes](@id common-breaking-changes-v4v2)

  - Use of termination conditions from `DiffEqBase` has been removed. Use the termination
    conditions from `NonlinearSolveBase` instead.
  - If no autodiff is provided, we now choose from a list of autodiffs based on the packages
    loaded. For example, if `Enzyme` is loaded, we will default to that (for reverse mode).
    In general, we don't guarantee the exact autodiff selected if `autodiff` is not provided
    (i.e. `nothing`).

## Dec '23

### Breaking Changes in `NonlinearSolve.jl` v3

  - `GeneralBroyden` and `GeneralKlement` have been renamed to `Broyden` and `Klement`
    respectively.
  - Compat for `SimpleNonlinearSolve` has been bumped to `v1`.
  - The old style of specifying autodiff with `chunksize`, `standardtag`, etc. has been
    deprecated in favor of directly specifying the autodiff type, like `AutoForwardDiff`.

### Breaking Changes in `SimpleNonlinearSolve.jl` v1

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
