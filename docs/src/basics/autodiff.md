# Automatic Differentiation Backends

## Summary of Finite Differencing Backends

  - [`AutoFiniteDiff`](@ref): Finite differencing, not optimal but always applicable.

## Summary of Forward Mode AD Backends

  - [`AutoForwardDiff`](@ref): The best choice for dense problems.
  - [`AutoPolyesterForwardDiff`](@ref): Might be faster than [`AutoForwardDiff`](@ref) for
    large problems. Requires `PolyesterForwardDiff.jl` to be installed and loaded.

## Summary of Reverse Mode AD Backends

  - [`AutoZygote`](@ref): The fastest choice for non-mutating array-based (BLAS) functions.
  - [`AutoEnzyme`](@ref): Uses `Enzyme.jl` Reverse Mode and should be considered
    experimental.

!!! note
    
    If `PolyesterForwardDiff.jl` is installed and loaded, then `SimpleNonlinearSolve.jl`
    will automatically use `AutoPolyesterForwardDiff` as the default AD backend.

!!! note
    
    The sparse versions of the methods refer to automated sparsity detection. These
    methods automatically discover the sparse Jacobian form from the function `f`. Note that
    all methods specialize the differentiation on a sparse Jacobian if the sparse Jacobian
    is given as `prob.f.jac_prototype` in the `NonlinearFunction` definition, and the
    `AutoSparse` here simply refers to whether this `jac_prototype` should be generated
    automatically. For more details, see
    [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl) and
    [Sparsity Detection Manual Entry](@ref sparsity-detection), as well as the
    documentation of [ADTypes.jl](https://github.com/SciML/ADTypes.jl).

## API Reference

```@docs
AutoSparse
```

### Finite Differencing Backends

```@docs
AutoFiniteDiff
```

### Forward Mode AD Backends

```@docs
AutoForwardDiff
AutoPolyesterForwardDiff
```

### Reverse Mode AD Backends

```@docs
AutoZygote
AutoEnzyme
```
