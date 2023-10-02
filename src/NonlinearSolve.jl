module NonlinearSolve

if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end

import PrecompileTools

PrecompileTools.@recompile_invalidations begin
    using DiffEqBase, LinearAlgebra, LinearSolve, SparseDiffTools
    using ADTypes, LineSearches, SciMLBase, SimpleNonlinearSolve
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
end

@reexport using ADTypes, LineSearches, SciMLBase, SimpleNonlinearSolve

const AbstractSparseADType = Union{ADTypes.AbstractSparseFiniteDifferences,
    ADTypes.AbstractSparseForwardMode, ADTypes.AbstractSparseReverseMode}

abstract type AbstractNonlinearSolveAlgorithm <: AbstractNonlinearAlgorithm end
abstract type AbstractNewtonAlgorithm{CJ, AD} <: AbstractNonlinearSolveAlgorithm end

function SciMLBase.__solve(prob::NonlinearProblem, alg::AbstractNonlinearSolveAlgorithm,
    args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

include("utils.jl")
include("linesearch.jl")
include("raphson.jl")
include("trustRegion.jl")
include("levenberg.jl")
include("jacobian.jl")
include("ad.jl")

# https://github.com/SciML/NonlinearSolve.jl/issues/223
@static if VERSION ≥ v"1.10-beta2"
    PrecompileTools.@compile_workload begin
        for T in (Float32, Float64)
            prob = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))

            precompile_algs = NewtonRaphson(), TrustRegion(), LevenbergMarquardt()

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
end

export RadiusUpdateSchemes

export NewtonRaphson, TrustRegion, LevenbergMarquardt

export LineSearch

end # module
