module NonlinearSolve

if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end

import Reexport: @reexport
import PrecompileTools: @recompile_invalidations, @compile_workload, @setup_workload

@recompile_invalidations begin
    using DiffEqBase, LinearAlgebra, LinearSolve, SparseArrays, SparseDiffTools
    using FastBroadcast: @..
    import ArrayInterface: restructure

    import ADTypes: AbstractFiniteDifferencesMode
    import ArrayInterface: undefmatrix,
        matrix_colors, parameterless_type, ismutable, issingular, fast_scalar_indexing
    import ConcreteStructs: @concrete
    import EnumX: @enumx
    import ForwardDiff
    import ForwardDiff: Dual
    import LinearSolve: ComposePreconditioner, InvPreconditioner, needs_concrete_A
    import RecursiveArrayTools: ArrayPartition,
        AbstractVectorOfArray, recursivecopy!, recursivefill!
    import SciMLBase: AbstractNonlinearAlgorithm, NLStats, _unwrap_val, has_jac, isinplace
    import StaticArraysCore: StaticArray, SVector, SArray, MArray
    import UnPack: @unpack

    using ADTypes, LineSearches, SciMLBase, SimpleNonlinearSolve
end

@reexport using ADTypes, LineSearches, SciMLBase, SimpleNonlinearSolve
import DiffEqBase: AbstractNonlinearTerminationMode,
    AbstractSafeNonlinearTerminationMode, AbstractSafeBestNonlinearTerminationMode,
    NonlinearSafeTerminationReturnCode, get_termination_mode

const AbstractSparseADType = Union{ADTypes.AbstractSparseFiniteDifferences,
    ADTypes.AbstractSparseForwardMode, ADTypes.AbstractSparseReverseMode}

abstract type AbstractNonlinearSolveLineSearchAlgorithm end

abstract type AbstractNonlinearSolveAlgorithm <: AbstractNonlinearAlgorithm end
abstract type AbstractNewtonAlgorithm{CJ, AD} <: AbstractNonlinearSolveAlgorithm end

abstract type AbstractNonlinearSolveCache{iip} end

isinplace(::AbstractNonlinearSolveCache{iip}) where {iip} = iip

function SciMLBase.__solve(prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem},
        alg::AbstractNonlinearSolveAlgorithm, args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

function not_terminated(cache::AbstractNonlinearSolveCache)
    return !cache.force_stop && cache.stats.nsteps < cache.maxiters
end
get_fu(cache::AbstractNonlinearSolveCache) = cache.fu1
set_fu!(cache::AbstractNonlinearSolveCache, fu) = (cache.fu1 = fu)
get_u(cache::AbstractNonlinearSolveCache) = cache.u
SciMLBase.set_u!(cache::AbstractNonlinearSolveCache, u) = (cache.u = u)

function SciMLBase.solve!(cache::AbstractNonlinearSolveCache)
    while not_terminated(cache)
        perform_step!(cache)
        cache.stats.nsteps += 1
    end

    # The solver might have set a different `retcode`
    if cache.retcode == ReturnCode.Default
        if cache.stats.nsteps == cache.maxiters
            cache.retcode = ReturnCode.MaxIters
        else
            cache.retcode = ReturnCode.Success
        end
    end

    return SciMLBase.build_solution(cache.prob, cache.alg, get_u(cache), get_fu(cache);
        cache.retcode, cache.stats)
end

include("utils.jl")
include("extension_algs.jl")
include("linesearch.jl")
include("raphson.jl")
include("trustRegion.jl")
include("levenberg.jl")
include("gaussnewton.jl")
include("dfsane.jl")
include("pseudotransient.jl")
include("broyden.jl")
include("klement.jl")
include("lbroyden.jl")
include("jacobian.jl")
include("ad.jl")
include("default.jl")

@setup_workload begin
    nlfuncs = ((NonlinearFunction{false}((u, p) -> u .* u .- p), 0.1),
        (NonlinearFunction{false}((u, p) -> u .* u .- p), [0.1]),
        (NonlinearFunction{true}((du, u, p) -> du .= u .* u .- p), [0.1]))
    probs_nls = NonlinearProblem[]
    for T in (Float32, Float64), (fn, u0) in nlfuncs
        push!(probs_nls, NonlinearProblem(fn, T.(u0), T(2)))
    end

    nls_algs = (NewtonRaphson(), TrustRegion(), LevenbergMarquardt(), PseudoTransient(),
        GeneralBroyden(), GeneralKlement(), DFSane(), nothing)

    probs_nlls = NonlinearLeastSquaresProblem[]
    nlfuncs = ((NonlinearFunction{false}((u, p) -> (u .^ 2 .- p)[1:1]), [0.1, 0.0]),
        (NonlinearFunction{false}((u, p) -> vcat(u .* u .- p, u .* u .- p)), [0.1, 0.1]),
        (NonlinearFunction{true}((du, u, p) -> du[1] = u[1] * u[1] - p,
                resid_prototype = zeros(1)), [0.1, 0.0]),
        (NonlinearFunction{true}((du, u, p) -> du .= vcat(u .* u .- p, u .* u .- p),
                resid_prototype = zeros(4)), [0.1, 0.1]))
    for (fn, u0) in nlfuncs
        push!(probs_nlls, NonlinearLeastSquaresProblem(fn, u0, 2.0))
    end
    nlfuncs = ((NonlinearFunction{false}((u, p) -> (u .^ 2 .- p)[1:1]), Float32[0.1, 0.0]),
        (NonlinearFunction{false}((u, p) -> vcat(u .* u .- p, u .* u .- p)),
            Float32[0.1, 0.1]),
        (NonlinearFunction{true}((du, u, p) -> du[1] = u[1] * u[1] - p,
                resid_prototype = zeros(Float32, 1)), Float32[0.1, 0.0]),
        (NonlinearFunction{true}((du, u, p) -> du .= vcat(u .* u .- p, u .* u .- p),
                resid_prototype = zeros(Float32, 4)), Float32[0.1, 0.1]))
    for (fn, u0) in nlfuncs
        push!(probs_nlls, NonlinearLeastSquaresProblem(fn, u0, 2.0f0))
    end

    nlls_algs = (LevenbergMarquardt(), GaussNewton(),
        LevenbergMarquardt(; linsolve = LUFactorization()),
        GaussNewton(; linsolve = LUFactorization()))

    @compile_workload begin
        for prob in probs_nls, alg in nls_algs
            solve(prob, alg, abstol = 1e-2)
        end
        for prob in probs_nlls, alg in nlls_algs
            solve(prob, alg, abstol = 1e-2)
        end
    end
end

export RadiusUpdateSchemes

export NewtonRaphson, TrustRegion, LevenbergMarquardt, DFSane, GaussNewton, PseudoTransient,
    GeneralBroyden, GeneralKlement, LimitedMemoryBroyden
export LeastSquaresOptimJL, FastLevenbergMarquardtJL
export NonlinearSolvePolyAlgorithm,
    RobustMultiNewton, FastShortcutNonlinearPolyalg, FastShortcutNLLSPolyalg

export LineSearch, LiFukushimaLineSearch

# Export the termination conditions from DiffEqBase
export SteadyStateDiffEqTerminationMode, SimpleNonlinearSolveTerminationMode,
    NormTerminationMode, RelTerminationMode, RelNormTerminationMode, AbsTerminationMode,
    AbsNormTerminationMode, RelSafeTerminationMode, AbsSafeTerminationMode,
    RelSafeBestTerminationMode, AbsSafeBestTerminationMode

end # module
