module NonlinearSolveHomotopyContinuation

using SciMLBase: AbstractNonlinearProblem
using SciMLBase
using NonlinearSolveBase
using SymbolicIndexingInterface
using LinearAlgebra
using ADTypes
import CommonSolve
import HomotopyContinuation as HC
import DifferentiationInterface as DI

using ConcreteStructs: @concrete

export HomotopyContinuationJL, HomotopyNonlinearFunction

@concrete struct HomotopyContinuationJL{AllRoots} <: NonlinearSolveBase.AbstractNonlinearSolveAlgorithm
    autodiff
    kwargs
end

function HomotopyContinuationJL{AllRoots}(; autodiff = true, kwargs...) where {AllRoots}
    if autodiff isa Bool
        autodiff = autodiff ? AutoForwardDiff() : AutoFiniteDiff()
    end
    HomotopyContinuationJL{AllRoots}(autodiff, kwargs)
end

HomotopyContinuationJL(; kwargs...) = HomotopyContinuationJL{false}(; kwargs...)

include("interface_types.jl")
include("solve.jl")

end
