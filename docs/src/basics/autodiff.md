# Automatic Differentiation Backends

## Summary of Finite Differencing Backends

  - [`AutoFiniteDiff`](@ref): Finite differencing, not optimal but always applicable.
  - [`AutoSparseFiniteDiff`](@ref): Sparse version of [`AutoFiniteDiff`](@ref).

## Summary of Forward Mode AD Backends

  - [`AutoForwardDiff`](@ref): The best choice for dense problems.
  - [`AutoSparseForwardDiff`](@ref): Sparse version of [`AutoForwardDiff`](@ref).
  - [`AutoPolyesterForwardDiff`](@ref): Might be faster than [`AutoForwardDiff`](@ref) for
    large problems. Requires `PolyesterForwardDiff.jl` to be installed and loaded.

## Summary of Reverse Mode AD Backends

  - [`AutoZygote`](@ref): The fastest choice for non-mutating array-based (BLAS) functions.
  - [`AutoSparseZygote`](@ref): Sparse version of [`AutoZygote`](@ref).
  - [`AutoEnzyme`](@ref): Uses `Enzyme.jl` Reverse Mode and should be considered
    experimental.

!!! note
    
    If `PolyesterForwardDiff.jl` is installed and loaded, then `SimpleNonlinearSolve.jl`
    will automatically use `AutoPolyesterForwardDiff` as the default AD backend.

## API Reference

### Finite Differencing Backends

```@docs
AutoFiniteDiff
AutoSparseFiniteDiff
```

### Forward Mode AD Backends

```@docs
AutoForwardDiff
AutoSparseForwardDiff
AutoPolyesterForwardDiff
```

### Reverse Mode AD Backends

```@docs
AutoZygote
AutoSparseZygote
AutoEnzyme
NonlinearSolve.AutoSparseEnzyme
```
