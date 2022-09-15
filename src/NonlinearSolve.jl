module NonlinearSolve

using Reexport
using UnPack: @unpack
using FiniteDiff, ForwardDiff
using ForwardDiff: Dual
using Setfield
using StaticArrays
using RecursiveArrayTools
using LinearAlgebra
import ArrayInterfaceCore
import IterativeSolvers
import RecursiveFactorization

@reexport using SciMLBase

abstract type AbstractNonlinearSolveAlgorithm <: SciMLBase.AbstractNonlinearAlgorithm end
abstract type AbstractBracketingAlgorithm <: AbstractNonlinearSolveAlgorithm end
abstract type AbstractNewtonAlgorithm{CS, AD, FDT, ST, CJ} <: AbstractNonlinearSolveAlgorithm end
abstract type AbstractImmutableNonlinearSolver <: AbstractNonlinearSolveAlgorithm end

include("utils.jl")
include("jacobian.jl")
include("types.jl")
include("solve.jl")
include("bisection.jl")
include("falsi.jl")
include("raphson.jl")
include("scalar.jl")

# DiffEq styled algorithms
export Bisection, Falsi, NewtonRaphson

end # module
