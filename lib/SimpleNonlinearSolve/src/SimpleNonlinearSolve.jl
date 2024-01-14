module SimpleNonlinearSolve

import PrecompileTools: @compile_workload, @setup_workload, @recompile_invalidations

@recompile_invalidations begin
    using ADTypes, ArrayInterface, ConcreteStructs, DiffEqBase, FastClosures, FiniteDiff,
        ForwardDiff, Reexport, LinearAlgebra, SciMLBase

    import DiffEqBase: AbstractNonlinearTerminationMode,
        AbstractSafeNonlinearTerminationMode, AbstractSafeBestNonlinearTerminationMode,
        NonlinearSafeTerminationReturnCode, get_termination_mode,
        NONLINEARSOLVE_DEFAULT_NORM
    import ForwardDiff: Dual
    import MaybeInplace: @bb, setindex_trait, CanSetindex, CannotSetindex
    import SciMLBase: AbstractNonlinearAlgorithm, build_solution, isinplace, _unwrap_val
    import StaticArraysCore: StaticArray, SVector, SMatrix, SArray, MArray, MMatrix, Size
end

@reexport using ADTypes, SciMLBase

abstract type AbstractSimpleNonlinearSolveAlgorithm <: AbstractNonlinearAlgorithm end
abstract type AbstractBracketingAlgorithm <: AbstractSimpleNonlinearSolveAlgorithm end
abstract type AbstractNewtonAlgorithm <: AbstractSimpleNonlinearSolveAlgorithm end

@inline __is_extension_loaded(::Val) = false

include("utils.jl")
include("linesearch.jl")

## Nonlinear Solvers
include("nlsolve/raphson.jl")
include("nlsolve/broyden.jl")
include("nlsolve/lbroyden.jl")
include("nlsolve/klement.jl")
include("nlsolve/trustRegion.jl")
include("nlsolve/halley.jl")
include("nlsolve/dfsane.jl")

## Interval Nonlinear Solvers
include("bracketing/bisection.jl")
include("bracketing/falsi.jl")
include("bracketing/ridder.jl")
include("bracketing/brent.jl")
include("bracketing/alefeld.jl")
include("bracketing/itp.jl")

# AD
include("ad.jl")

## Default algorithm

# Set the default bracketing method to ITP
function SciMLBase.solve(prob::IntervalNonlinearProblem; kwargs...)
    return solve(prob, ITP(); kwargs...)
end

function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::Nothing,
        args...; kwargs...)
    return solve(prob, ITP(), args...; kwargs...)
end

# By Pass the highlevel checks for NonlinearProblem for Simple Algorithms
function SciMLBase.solve(prob::NonlinearProblem, alg::AbstractSimpleNonlinearSolveAlgorithm,
        args...; kwargs...)
    return SciMLBase.__solve(prob, alg, args...; kwargs...)
end

function SciMLBase.solve(prob::NonlinearProblem{<:Union{<:Number, <:SArray}},
        alg::AbstractSimpleNonlinearSolveAlgorithm, args...; abstol = nothing,
        reltol = nothing, kwargs...)
    _abstol = __get_tolerance(prob.u0, abstol, eltype(prob.u0))
    _reltol = __get_tolerance(prob.u0, reltol, eltype(prob.u0))
    return SciMLBase.__solve(prob, alg, args...; abstol = _abstol, reltol = _reltol,
        kwargs...)
end

@setup_workload begin
    for T in (Float32, Float64)
        prob_no_brack_scalar = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))
        prob_no_brack_iip = NonlinearProblem{true}((du, u, p) -> du .= u .* u .- p,
            T.([1.0, 1.0, 1.0]), T(2))
        prob_no_brack_oop = NonlinearProblem{false}((u, p) -> u .* u .- p,
            T.([1.0, 1.0, 1.0]), T(2))

        algs = [SimpleNewtonRaphson(), SimpleBroyden(), SimpleKlement(), SimpleDFSane(),
            SimpleTrustRegion(), SimpleLimitedMemoryBroyden(; threshold = 2)]

        algs_no_iip = [SimpleHalley()]

        @compile_workload begin
            for alg in algs
                solve(prob_no_brack_scalar, alg, abstol = T(1e-2))
                solve(prob_no_brack_iip, alg, abstol = T(1e-2))
                solve(prob_no_brack_oop, alg, abstol = T(1e-2))
            end

            for alg in algs_no_iip
                solve(prob_no_brack_scalar, alg, abstol = T(1e-2))
                solve(prob_no_brack_oop, alg, abstol = T(1e-2))
            end
        end

        prob_brack = IntervalNonlinearProblem{false}((u, p) -> u * u - p,
            T.((0.0, 2.0)), T(2))
        algs = [Bisection(), Falsi(), Ridder(), Brent(), Alefeld(), ITP()]
        @compile_workload begin
            for alg in algs
                solve(prob_brack, alg, abstol = T(1e-2))
            end
        end
    end
end

export SimpleBroyden, SimpleDFSane, SimpleGaussNewton, SimpleHalley, SimpleKlement,
    SimpleLimitedMemoryBroyden, SimpleNewtonRaphson, SimpleTrustRegion
export Alefeld, Bisection, Brent, Falsi, ITP, Ridder

end # module
