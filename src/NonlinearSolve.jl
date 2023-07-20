module NonlinearSolve
if isdefined(Base, :Experimental) &&
   isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end
using Reexport
using UnPack: @unpack
using FiniteDiff, ForwardDiff
using ForwardDiff: Dual
using LinearAlgebra
using StaticArraysCore
using RecursiveArrayTools
import EnumX
import ArrayInterface
import LinearSolve
using DiffEqBase
using SparseDiffTools
using LineSearches

@reexport using SciMLBase
using SciMLBase: NLStats
@reexport using SimpleNonlinearSolve

import SciMLBase: _unwrap_val

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
include("trustRegion.jl")
include("levenberg.jl")
include("ad.jl")

import PrecompileTools

PrecompileTools.@compile_workload begin
    for T in (Float32, Float64)
        prob = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))

        precompile_algs = if VERSION >= v"1.7"
            (NewtonRaphson(), TrustRegion(), LevenbergMarquardt())
        else
            (NewtonRaphson(),)
        end

        for alg in precompile_algs
            solve(prob, alg, abstol = T(1e-2))
        end

        prob = NonlinearProblem{true}((du, u, p) -> du[1] = u[1] * u[1] - p[1], T[0.1],
            T[2])
        for alg in precompile_algs
            solve(prob, alg, abstol = T(1e-2))
        end
    end
end

export RadiusUpdateSchemes
export LineSearches
export NewtonRaphson, TrustRegion, LevenbergMarquardt


end # module
