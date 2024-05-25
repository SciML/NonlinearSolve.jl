module SimpleNonlinearSolve

using PrecompileTools: @compile_workload, @setup_workload, @recompile_invalidations

@recompile_invalidations begin
    using ADTypes: ADTypes, AutoFiniteDiff, AutoForwardDiff, AutoPolyesterForwardDiff
    using ArrayInterface: ArrayInterface
    using ConcreteStructs: @concrete
    using DiffEqBase: DiffEqBase, AbstractNonlinearTerminationMode,
                      AbstractSafeNonlinearTerminationMode,
                      AbstractSafeBestNonlinearTerminationMode, AbsNormTerminationMode,
                      NONLINEARSOLVE_DEFAULT_NORM
    using DiffResults: DiffResults
    using FastClosures: @closure
    using FiniteDiff: FiniteDiff
    using ForwardDiff: ForwardDiff, Dual
    using LinearAlgebra: LinearAlgebra, I, convert, copyto!, diagind, dot, issuccess, lu,
                         mul!, norm, transpose
    using MaybeInplace: @bb, setindex_trait, CanSetindex, CannotSetindex
    using Reexport: @reexport
    using SciMLBase: SciMLBase, IntervalNonlinearProblem, NonlinearFunction,
                     NonlinearLeastSquaresProblem, NonlinearProblem, ReturnCode, init,
                     remake, solve, AbstractNonlinearAlgorithm, build_solution, isinplace,
                     _unwrap_val
    using StaticArraysCore: StaticArray, SVector, SMatrix, SArray, MArray, Size
end

@reexport using SciMLBase

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
SciMLBase.solve(prob::IntervalNonlinearProblem; kwargs...) = solve(prob, ITP(); kwargs...)
function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::Nothing, args...; kwargs...)
    return solve(prob, ITP(), args...; prob.kwargs..., kwargs...)
end

# By Pass the highlevel checks for NonlinearProblem for Simple Algorithms
function SciMLBase.solve(prob::NonlinearProblem, alg::AbstractSimpleNonlinearSolveAlgorithm,
        args...; sensealg = nothing, u0 = nothing, p = nothing, kwargs...)
    if sensealg === nothing && haskey(prob.kwargs, :sensealg)
        sensealg = prob.kwargs[:sensealg]
    end
    new_u0 = u0 !== nothing ? u0 : prob.u0
    new_p = p !== nothing ? p : prob.p
    return __internal_solve_up(prob, sensealg, new_u0, u0 === nothing, new_p,
        p === nothing, alg, args...; prob.kwargs..., kwargs...)
end

function __internal_solve_up(_prob::NonlinearProblem, sensealg, u0, u0_changed,
        p, p_changed, alg, args...; kwargs...)
    prob = u0_changed || p_changed ? remake(_prob; u0, p) : _prob
    return SciMLBase.__solve(prob, alg, args...; kwargs...)
end

@setup_workload begin
    for T in (Float32, Float64)
        prob_no_brack_scalar = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))
        prob_no_brack_iip = NonlinearProblem{true}(
            (du, u, p) -> du .= u .* u .- p, T.([1.0, 1.0, 1.0]), T(2))
        prob_no_brack_oop = NonlinearProblem{false}(
            (u, p) -> u .* u .- p, T.([1.0, 1.0, 1.0]), T(2))

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

        prob_brack = IntervalNonlinearProblem{false}(
            (u, p) -> u * u - p, T.((0.0, 2.0)), T(2))
        algs = [Bisection(), Falsi(), Ridder(), Brent(), Alefeld(), ITP()]
        @compile_workload begin
            for alg in algs
                solve(prob_brack, alg, abstol = T(1e-2))
            end
        end
    end
end

export AutoFiniteDiff, AutoForwardDiff, AutoPolyesterForwardDiff
export SimpleBroyden, SimpleDFSane, SimpleGaussNewton, SimpleHalley, SimpleKlement,
       SimpleLimitedMemoryBroyden, SimpleNewtonRaphson, SimpleTrustRegion
export Alefeld, Bisection, Brent, Falsi, ITP, Ridder

end # module
