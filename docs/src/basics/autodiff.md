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

!!! note
    
    The `Sparse` versions of the methods refers to automated sparsity detection. These
    methods automatically discover the sparse Jacobian form from the function `f`. Note that
    all methods specialize the differentiation on a sparse Jacobian if the sparse Jacobian
    is given as `prob.f.jac_prototype` in the `NonlinearFunction` definition, and the
    `AutoSparse` here simply refers to whether this `jac_prototype` should be generated
    automatically. For more details, see
    [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl) and
    [Sparsity Detection Manual Entry](@ref sparsity-detection).

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
