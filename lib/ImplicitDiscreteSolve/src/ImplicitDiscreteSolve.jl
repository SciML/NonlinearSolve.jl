module ImplicitDiscreteSolve

using SciMLBase: AbstractNonlinearProblem
using SciMLBase
using NonlinearSolveBase
using SymbolicIndexingInterface
using LinearAlgebra
using ADTypes
using TaylorDiff
using DocStringExtensions
import CommonSolve
import DifferentiationInterface as DI

using ConcreteStructs: @concrete

"""
    IteratedNonlinearSolve(; nlsolvealg, autodiff = true, kwargs...)

This algorithm is a solver for ImplicitDiscreteSystems.
"""
@concrete struct IteratedNonlinearSolve <: NonlinearSolveBase.AbstractNonlinearSolveAlgorithm
    nlsolvealg
    autodiff
    kwargs
end

export IteratedNonlinearSolve

include("solve.jl")

end
