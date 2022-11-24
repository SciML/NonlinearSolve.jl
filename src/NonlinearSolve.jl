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
import LinearSolve
using DiffEqBase

@reexport using SciMLBase
@reexport using SimpleNonlinearSolve

abstract type AbstractNonlinearSolveAlgorithm <: SciMLBase.AbstractNonlinearAlgorithm end
abstract type AbstractNewtonAlgorithm{CS, AD, FDT, ST, CJ} <:
              AbstractNonlinearSolveAlgorithm end

function SciMLBase.__solve(prob::NonlinearProblem,
                           alg::AbstractNonlinearSolveAlgorithm, args...;
                           kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    sol = solve!(cache)
end

include("utils.jl")
include("jacobian.jl")
include("raphson.jl")
include("ad.jl")

export NewtonRaphson

end # module
