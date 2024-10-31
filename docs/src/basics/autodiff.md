# Automatic Differentiation Backends

!!! note
    
    We support all backends supported by DifferentiationInterface.jl. Please refer to
    the [backends page](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/explanation/backends/)
    for more information.

## Summary of Finite Differencing Backends

  - [`AutoFiniteDiff`](@extref ADTypes.AutoFiniteDiff): Finite differencing using
    `FiniteDiff.jl`, not optimal but always applicable.
  - [`AutoFiniteDifferences`](@extref ADTypes.AutoFiniteDifferences): Finite differencing
    using `FiniteDifferences.jl`, not optimal but always applicable.

## Summary of Forward Mode AD Backends

  - [`AutoForwardDiff`](@extref ADTypes.AutoForwardDiff): The best choice for dense
    problems.
  - [`AutoPolyesterForwardDiff`](@extref ADTypes.AutoPolyesterForwardDiff): Might be faster
    than [`AutoForwardDiff`](@extref ADTypes.AutoForwardDiff) for large problems. Requires
    `PolyesterForwardDiff.jl` to be installed and loaded.

## Summary of Reverse Mode AD Backends

  - [`AutoZygote`](@extref ADTypes.AutoZygote): The fastest choice for non-mutating
    array-based (BLAS) functions.
  - [`AutoEnzyme`](@extref ADTypes.AutoEnzyme): Uses `Enzyme.jl` Reverse Mode and works for
    both in-place and out-of-place functions.

!!! tip
    
    For sparsity detection and sparse AD take a look at
    [sparsity detection](@ref sparsity-detection).
