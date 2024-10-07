module SimpleNonlinearSolve

using Accessors: @reset
using CommonSolve: CommonSolve, solve
using ConcreteStructs: @concrete
using FastClosures: @closure
using LineSearch: LiFukushimaLineSearch
using LinearAlgebra: LinearAlgebra, dot
using MaybeInplace: @bb, setindex_trait, CannotSetindex, CanSetindex
using PrecompileTools: @compile_workload, @setup_workload
using Reexport: @reexport
@reexport using SciMLBase  # I don't like this but needed to avoid a breaking change
using SciMLBase: AbstractNonlinearAlgorithm, NonlinearProblem, ReturnCode
using StaticArraysCore: StaticArray, SArray, SVector, MArray

# AD Dependencies
using ADTypes: AutoFiniteDiff, AutoForwardDiff, AutoPolyesterForwardDiff
using DifferentiationInterface: DifferentiationInterface
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff

using BracketingNonlinearSolve: Alefeld, Bisection, Brent, Falsi, ITP, Ridder
using NonlinearSolveBase: NonlinearSolveBase, ImmutableNonlinearProblem, L2_NORM

const DI = DifferentiationInterface

abstract type AbstractSimpleNonlinearSolveAlgorithm <: AbstractNonlinearAlgorithm end

const safe_similar = NonlinearSolveBase.Utils.safe_similar

is_extension_loaded(::Val) = false

include("utils.jl")

include("broyden.jl")
include("dfsane.jl")
include("halley.jl")
include("klement.jl")
include("lbroyden.jl")
include("raphson.jl")
include("trust_region.jl")

# By Pass the highlevel checks for NonlinearProblem for Simple Algorithms
function CommonSolve.solve(prob::NonlinearProblem,
        alg::AbstractSimpleNonlinearSolveAlgorithm, args...; kwargs...)
    prob = convert(ImmutableNonlinearProblem, prob)
    return solve(prob, alg, args...; kwargs...)
end

function CommonSolve.solve(
        prob::ImmutableNonlinearProblem, alg::AbstractSimpleNonlinearSolveAlgorithm,
        args...; sensealg = nothing, u0 = nothing, p = nothing, kwargs...)
    if sensealg === nothing && haskey(prob.kwargs, :sensealg)
        sensealg = prob.kwargs[:sensealg]
    end
    new_u0 = u0 !== nothing ? u0 : prob.u0
    new_p = p !== nothing ? p : prob.p
    return simplenonlinearsolve_solve_up(prob, sensealg, new_u0, u0 === nothing, new_p,
        p === nothing, alg, args...; prob.kwargs..., kwargs...)
end

function simplenonlinearsolve_solve_up(
        prob::ImmutableNonlinearProblem, sensealg, u0, u0_changed, p, p_changed,
        alg::AbstractSimpleNonlinearSolveAlgorithm, args...; kwargs...)
    (u0_changed || p_changed) && (prob = remake(prob; u0, p))
    return SciMLBase.__solve(prob, alg, args...; kwargs...)
end

# NOTE: This is defined like this so that we don't have to keep have 2 args for the
# extensions
function solve_adjoint(args...; kws...)
    is_extension_loaded(Val(:DiffEqBase)) && return solve_adjoint_internal(args...; kws...)
    error("Adjoint sensitivity analysis requires `DiffEqBase.jl` to be explicitly loaded.")
end

function solve_adjoint_internal end

@setup_workload begin
    for T in (Float64,)
        prob_scalar = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))
        prob_iip = NonlinearProblem{true}((du, u, p) -> du .= u .* u .- p, ones(T, 3), T(2))
        prob_oop = NonlinearProblem{false}((u, p) -> u .* u .- p, ones(T, 3), T(2))

        # Only compile frequently used algorithms -- mostly from the NonlinearSolve default
        #!format: off
        algs = [
            SimpleBroyden(),
            # SimpleDFSane(),
            SimpleKlement(),
            # SimpleLimitedMemoryBroyden(),
            # SimpleHalley(),
            SimpleNewtonRaphson(; autodiff = AutoForwardDiff(; chunksize = 1)),
            # SimpleTrustRegion()
        ]
        #!format: on

        @compile_workload begin
            @sync for alg in algs
                for prob in (prob_scalar, prob_iip, prob_oop)
                    Threads.@spawn CommonSolve.solve(prob, alg; abstol = 1e-2)
                end
            end
        end
    end
end

export AutoFiniteDiff, AutoForwardDiff, AutoPolyesterForwardDiff

export Alefeld, Bisection, Brent, Falsi, ITP, Ridder

export SimpleBroyden, SimpleKlement, SimpleLimitedMemoryBroyden
export SimpleDFSane
export SimpleGaussNewton, SimpleNewtonRaphson, SimpleTrustRegion
export SimpleHalley

end
