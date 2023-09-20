module NonlinearSolve

if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end

using DiffEqBase, LinearAlgebra, LinearSolve, SparseDiffTools
import ForwardDiff

import ADTypes: AbstractFiniteDifferencesMode
import ArrayInterface: undefmatrix, matrix_colors, parameterless_type, ismutable
import ConcreteStructs: @concrete
import EnumX: @enumx
import ForwardDiff: Dual
import LinearSolve: ComposePreconditioner, InvPreconditioner, needs_concrete_A
import RecursiveArrayTools: ArrayPartition,
    AbstractVectorOfArray, recursivecopy!, recursivefill!
import Reexport: @reexport
import SciMLBase: AbstractNonlinearAlgorithm, NLStats, _unwrap_val, has_jac, isinplace
import StaticArraysCore: StaticArray, SVector, SArray, MArray
import UnPack: @unpack

@reexport using ADTypes, LineSearches, SciMLBase, SimpleNonlinearSolve

const AbstractSparseADType = Union{ADTypes.AbstractSparseFiniteDifferences,
    ADTypes.AbstractSparseForwardMode, ADTypes.AbstractSparseReverseMode}

abstract type AbstractNonlinearSolveAlgorithm <: AbstractNonlinearAlgorithm end
abstract type AbstractNewtonAlgorithm{CJ, AD} <: AbstractNonlinearSolveAlgorithm end

abstract type AbstractNonlinearSolveCache{iip} end

isinplace(::AbstractNonlinearSolveCache{iip}) where {iip} = iip

function SciMLBase.__solve(prob::NonlinearProblem, alg::AbstractNonlinearSolveAlgorithm,
    args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

function SciMLBase.solve!(cache::AbstractNonlinearSolveCache)
    while !cache.force_stop && cache.stats.nsteps < cache.maxiters
        perform_step!(cache)
        cache.stats.nsteps += 1
    end

    if cache.stats.nsteps == cache.maxiters
        cache.retcode = ReturnCode.MaxIters
    else
        cache.retcode = ReturnCode.Success
    end

    return SciMLBase.build_solution(cache.prob, cache.alg, cache.u, cache.fu1;
        cache.retcode, cache.stats)
end

include("utils.jl")
include("linesearch.jl")
include("raphson.jl")
include("trustRegion.jl")
include("levenberg.jl")
include("gaussnewton.jl")
include("jacobian.jl")
include("ad.jl")

import PrecompileTools

PrecompileTools.@compile_workload begin
    for T in (Float32, Float64)
        prob = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))

        precompile_algs = if VERSION ≥ v"1.7"
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

export NewtonRaphson, TrustRegion, LevenbergMarquardt, GaussNewton

export LineSearch

end # module
