module SimpleNonlinearSolve

using ADTypes: ADTypes, AbstractADType, AutoFiniteDiff, AutoForwardDiff,
               AutoPolyesterForwardDiff
using CommonSolve: CommonSolve, solve
using PrecompileTools: @compile_workload, @setup_workload
using Reexport: @reexport
@reexport using SciMLBase  # I don't like this but needed to avoid a breaking change
using SciMLBase: AbstractNonlinearAlgorithm, NonlinearProblem, ReturnCode

using BracketingNonlinearSolve: Alefeld, Bisection, Brent, Falsi, ITP, Ridder
using NonlinearSolveBase: ImmutableNonlinearProblem

abstract type AbstractSimpleNonlinearSolveAlgorithm <: AbstractNonlinearAlgorithm end

is_extension_loaded(::Val) = false

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
    @compile_workload begin end
end

export AutoFiniteDiff, AutoForwardDiff, AutoPolyesterForwardDiff

export Alefeld, Bisection, Brent, Falsi, ITP, Ridder

end
