module NonlinearSolveHomotopyContinuation

using SciMLBase: AbstractNonlinearProblem
using SciMLBase
using NonlinearSolveBase
using SymbolicIndexingInterface
using LinearAlgebra
using ADTypes
using TaylorDiff
using DocStringExtensions
import CommonSolve
import HomotopyContinuation as HC
import DifferentiationInterface as DI

using ConcreteStructs: @concrete

export HomotopyContinuationJL, HomotopyNonlinearFunction

"""
    HomotopyContinuationJL{AllRoots, ComplexRoots}(; autodiff = true, kwargs...)
    HomotopyContinuationJL(; kwargs...) = HomotopyContinuationJL{false}(; kwargs...)

This algorithm is an interface to `HomotopyContinuation.jl`. It is only valid for
fully determined polynomial systems. The `AllRoots` type parameter can be `true` or
`false` and controls whether the solver will find all roots of the polynomial
or a single root close to the initial guess provided to the `NonlinearProblem`.
The `ComplexRoots` type parameter can be `Val{true}` or `Val{false}` (default) and 
controls whether complex roots are returned or filtered to only real roots.
The polynomial function must allow complex numbers to be provided as the state.

If `AllRoots` is `true`, the initial guess in the `NonlinearProblem` is ignored.
The function must be traceable using HomotopyContinuation.jl's symbolic variables.
Note that higher degree polynomials and systems with multiple unknowns can increase
solve time significantly.

If `AllRoots` is `false`, a single path is traced during the homotopy. The traced path
depends on the initial guess provided to the `NonlinearProblem` being solved. This method
does not require that the polynomial function is traceable via HomotopyContinuation.jl's
symbolic variables.

If `ComplexRoots` is `Val{true}`, complex roots will be returned. If `Val{false}` (default),
only real roots will be returned.

HomotopyContinuation.jl requires the jacobian of the system. In case a jacobian function
is provided, it will be used. Otherwise, the `autodiff` keyword argument controls the
autodiff method used to compute the jacobian. A value of `true` refers to
`AutoForwardDiff` and `false` refers to `AutoFiniteDiff`. Alternate algorithms can be
specified using ADTypes.jl.

HomotopyContinuation.jl requires the taylor series of the polynomial system for the single
root method. This is automatically computed using TaylorSeries.jl.
"""
@concrete struct HomotopyContinuationJL{AllRoots, ComplexRoots} <:
                 NonlinearSolveBase.AbstractNonlinearSolveAlgorithm
    autodiff
    kwargs
end

function HomotopyContinuationJL{AllRoots, ComplexRoots}(; autodiff = true, kwargs...) where {AllRoots, ComplexRoots}
    if autodiff isa Bool
        autodiff = autodiff ? AutoForwardDiff() : AutoFiniteDiff()
    end
    HomotopyContinuationJL{AllRoots, ComplexRoots}(autodiff, kwargs)
end

function HomotopyContinuationJL{AllRoots}(; autodiff = true, kwargs...) where {AllRoots}
    HomotopyContinuationJL{AllRoots, Val{false}}(; autodiff, kwargs...)
end

HomotopyContinuationJL(; kwargs...) = HomotopyContinuationJL{false}(; kwargs...)

function HomotopyContinuationJL(alg::HomotopyContinuationJL{R, C}; kwargs...) where {R, C}
    HomotopyContinuationJL{R, C}(; autodiff = alg.autodiff, alg.kwargs..., kwargs...)
end

include("interface_types.jl")
include("jacobian_handling.jl")
include("solve.jl")

end
