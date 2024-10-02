"""
    SimpleHouseholder{order}()

A low-overhead implementation of Householder's method. This method is non-allocating on scalar
and static array problems.

Internally, this uses TaylorDiff.jl for the automatic differentiation.

### Type Parameters

  - `order`: the convergence order of the Householder method. `order = 2` is the same as Newton's method, `order = 3` is the same as Halley's method, etc.
"""
struct SimpleHouseholder{order} <: AbstractNewtonAlgorithm end
