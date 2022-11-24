module SimpleNonlinearSolve

using Reexport
using FiniteDiff, ForwardDiff
using ForwardDiff: Dual
using StaticArraysCore
import ArrayInterfaceCore

@reexport using SciMLBase

abstract type AbstractSimpleNonlinearSolveAlgorithm <: SciMLBase.AbstractNonlinearAlgorithm end
abstract type AbstractBracketingAlgorithm <: AbstractSimpleNonlinearSolveAlgorithm end
abstract type AbstractNewtonAlgorithm{CS, AD, FDT} <: AbstractSimpleNonlinearSolveAlgorithm end
abstract type AbstractImmutableNonlinearSolver <: AbstractSimpleNonlinearSolveAlgorithm end

include("utils.jl")
include("bisection.jl")
include("falsi.jl")
include("raphson.jl")
include("ad.jl")

# DiffEq styled algorithms
export Bisection, Falsi, SimpleNewtonRaphson

end # module
