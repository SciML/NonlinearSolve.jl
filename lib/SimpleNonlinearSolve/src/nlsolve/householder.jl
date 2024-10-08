"""
    SimpleHouseholder{order}()

A low-overhead implementation of Householder's method to arbitrary order.
This method is non-allocating on scalar and static array problems.

!!! warning

    Needs `TaylorDiff.jl` to be explicitly loaded before using this functionality.
    Internally, this uses TaylorDiff.jl for automatic differentiation.

### Type Parameters

  - `order`: the convergence order of the Householder method. `order = 2` is the same as Newton's method, `order = 3` is the same as Halley's method, etc.
"""
struct SimpleHouseholder{order} <: AbstractNewtonAlgorithm end
