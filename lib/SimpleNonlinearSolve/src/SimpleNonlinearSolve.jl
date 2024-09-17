module SimpleNonlinearSolve

using CommonSolve: CommonSolve, solve
using FastClosures: @closure
using MaybeInplace: @bb
using PrecompileTools: @compile_workload, @setup_workload
using Reexport: @reexport
@reexport using SciMLBase  # I don't like this but needed to avoid a breaking change
using SciMLBase: AbstractNonlinearAlgorithm, NonlinearProblem, ReturnCode
using StaticArraysCore: StaticArray

# AD Dependencies
using ADTypes: AbstractADType, AutoFiniteDiff, AutoForwardDiff, AutoPolyesterForwardDiff
using DifferentiationInterface: DifferentiationInterface
# TODO: move these to extensions in a breaking change. These are not even used in the
#       package, but are used to trigger the extension loading in DI.jl
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff

using BracketingNonlinearSolve: Alefeld, Bisection, Brent, Falsi, ITP, Ridder
using NonlinearSolveBase: ImmutableNonlinearProblem, get_tolerance

const DI = DifferentiationInterface

abstract type AbstractSimpleNonlinearSolveAlgorithm <: AbstractNonlinearAlgorithm end

is_extension_loaded(::Val) = false

include("utils.jl")

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
    for T in (Float32, Float64)
        prob_scalar = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))
        prob_iip = NonlinearProblem{true}((du, u, p) -> du .= u .* u .- p, ones(T, 3), T(2))
        prob_oop = NonlinearProblem{false}((u, p) -> u .* u .- p, ones(T, 3), T(2))

        algs = []
        algs_no_iip = []

        @compile_workload begin
            for alg in algs, prob in (prob_scalar, prob_iip, prob_oop)
                CommonSolve.solve(prob, alg)
            end
            for alg in algs_no_iip
                CommonSolve.solve(prob_scalar, alg)
            end
        end
    end
end

export AutoFiniteDiff, AutoForwardDiff, AutoPolyesterForwardDiff

export Alefeld, Bisection, Brent, Falsi, ITP, Ridder

end
