# This just documents the AD types from ADTypes.jl

"""
    AutoFiniteDiff(; fdtype = Val(:forward), fdjtype = fdtype, fdhtype = Val(:hcentral))

This uses [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl). While not necessarily
the most efficient, this is the only choice that doesn't require the `f` function to be
automatically differentiable, which means it applies to any choice. However, because it's
using finite differencing, one needs to be careful as this procedure introduces numerical
error into the derivative estimates.

  - Compatible with GPUs
  - Can be used for Jacobian-Vector Products (JVPs)
  - Can be used for Vector-Jacobian Products (VJPs)
  - Supports both inplace and out-of-place functions

### Keyword Arguments

  - `fdtype`: the method used for defining the gradient
  - `fdjtype`: the method used for defining the Jacobian of constraints.
  - `fdhtype`: the method used for defining the Hessian
"""
AutoFiniteDiff

"""
    AutoForwardDiff(; chunksize = nothing, tag = nothing)
    AutoForwardDiff{chunksize, tagType}(tag::tagType)

This uses the [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) package. It is
the fastest choice for square or wide systems. It is easy to use and compatible with most
Julia functions which have loose type restrictions.

  - Compatible with GPUs
  - Can be used for Jacobian-Vector Products (JVPs)
  - Supports both inplace and out-of-place functions

For type-stability of internal operations, a positive `chunksize` must be provided.

### Keyword Arguments

  - `chunksize`: Count of dual numbers that can be propagated simultaneously. Setting this
    number to a high value will lead to slowdowns. Use
    [`NonlinearSolve.pickchunksize`](@ref) to get a proper value.
  - `tag`: Used to avoid perturbation confusion. If set to `nothing`, we use a custom tag.
"""
AutoForwardDiff

"""
    AutoPolyesterForwardDiff(; chunksize = nothing)

Uses [`PolyesterForwardDiff.jl`](https://github.com/JuliaDiff/PolyesterForwardDiff.jl)
to compute the jacobian. This is essentially parallelized `ForwardDiff.jl`.

  - Supports both inplace and out-of-place functions

### Keyword Arguments

  - `chunksize`: Count of dual numbers that can be propagated simultaneously. Setting
    this number to a high value will lead to slowdowns. Use
    [`NonlinearSolve.pickchunksize`](@ref) to get a proper value.
"""
AutoPolyesterForwardDiff

"""
    AutoZygote()

Uses [`Zygote.jl`](https://github.com/FluxML/Zygote.jl) package. This is the staple
reverse-mode AD that handles a large portion of Julia with good efficiency.

  - Compatible with GPUs
  - Can be used for Vector-Jacobian Products (VJPs)
  - Supports only out-of-place functions

For VJPs this is the current best choice. This is the most efficient method for long
jacobians.
"""
AutoZygote

"""
    AutoEnzyme()

Uses reverse mode [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl). This is currently
experimental, and not extensively tested on our end. We only support Jacobian construction
and VJP support is currently not implemented.

  - Supports both inplace and out-of-place functions
"""
AutoEnzyme

"""
    AutoSparse(AutoEnzyme())

Sparse version of [`AutoEnzyme`](@ref) that uses
[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) and the row color vector of
the Jacobian Matrix to efficiently compute the Sparse Jacobian.

  - Supports both inplace and out-of-place functions

This is efficient only for long jacobians or if the maximum value of the row color vector is
significantly lower than the maximum value of the column color vector.

    AutoSparse(AutoFiniteDiff())

Sparse Version of [`AutoFiniteDiff`](@ref) that uses
[FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl) and the column color vector of
the Jacobian Matrix to efficiently compute the Sparse Jacobian.

  - Supports both inplace and out-of-place functions

    AutoSparse(AutoForwardDiff(; chunksize = nothing, tag = nothing))
    AutoSparse(AutoForwardDiff{chunksize, tagType}(tag::tagType))

Sparse Version of [`AutoForwardDiff`](@ref) that uses
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and the column color vector of
the Jacobian Matrix to efficiently compute the Sparse Jacobian.

  - Supports both inplace and out-of-place functions

For type-stability of internal operations, a positive `chunksize` must be provided.

### Keyword Arguments

  - `chunksize`: Count of dual numbers that can be propagated simultaneously. Setting this
    number to a high value will lead to slowdowns. Use
    [`NonlinearSolve.pickchunksize`](@ref) to get a proper value.

  - `tag`: Used to avoid perturbation confusion. If set to `nothing`, we use a custom tag.

    AutoSparse(AutoZygote())

Sparse version of [`AutoZygote`](@ref) that uses
[`Zygote.jl`](https://github.com/FluxML/Zygote.jl) and the row color vector of
the Jacobian Matrix to efficiently compute the Sparse Jacobian.

  - Supports only out-of-place functions

This is efficient only for long jacobians or if the maximum value of the row color vector is
significantly lower than the maximum value of the column color vector.
"""
AutoSparse
