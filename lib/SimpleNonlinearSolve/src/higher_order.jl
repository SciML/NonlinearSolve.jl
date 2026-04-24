"""
    SimpleHouseholder{order}()

A low-overhead implementation of Householder's method to arbitrary order. Only works for scalar problems.

!!! warning

    Needs `TaylorDiff.jl` to be explicitly loaded before using this functionality.
    Internally, this uses TaylorDiff.jl for automatic differentiation.

### Type Parameters

  - `P`: the order of the Householder method. `P = 1` is the same as Newton's method, `P = 2` is the same as Halley's method
"""
struct SimpleHouseholder{P} <: AbstractSimpleNonlinearSolveAlgorithm end

"""
    SimpleInverseTaylor(order, autodiff)
    SimpleInverseTaylor(; order = Val{2}(), autodiff = nothing)

A low-overhead implementation of the inverse Taylor method to arbitrary order.

!!! warning

    Needs `TaylorDiff.jl` to be explicitly loaded before using this functionality.
    Internally, this uses TaylorDiff.jl for automatic differentiation.

### Type Parameters
    - `P`: the order of the inverse Taylor method. `P = 1` is the same as Newton's method, `P = 2` is the same as Chebyshev's method
"""
@kwdef @concrete struct SimpleInverseTaylor <: AbstractSimpleNonlinearSolveAlgorithm
    order = Val(2)
    autodiff = nothing
end
