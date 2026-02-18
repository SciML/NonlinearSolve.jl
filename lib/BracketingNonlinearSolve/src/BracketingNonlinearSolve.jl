module BracketingNonlinearSolve

using ConcreteStructs: @concrete
using PrecompileTools: @compile_workload, @setup_workload
using Reexport: @reexport

using CommonSolve: CommonSolve, solve
using NonlinearSolveBase: NonlinearSolveBase, AbstractNonlinearSolveAlgorithm, NonlinearVerbosity, @SciMLMessage, AbstractVerbosityPreset
using SciMLBase: SciMLBase, IntervalNonlinearProblem, ReturnCode

abstract type AbstractBracketingAlgorithm <: AbstractNonlinearSolveAlgorithm end

include("utils.jl")
include("common.jl")

include("alefeld.jl")
include("bisection.jl")
include("brent.jl")
include("falsi.jl")
include("itp.jl")
include("muller.jl")
include("ridder.jl")
include("modAB.jl")

# Default Algorithm
function CommonSolve.solve(prob::IntervalNonlinearProblem; kwargs...)
    return CommonSolve.solve(prob, ModAB(); kwargs...)
end

function CommonSolve.solve(prob::IntervalNonlinearProblem, nothing, args...; kwargs...)
    return CommonSolve.solve(prob, ModAB(), args...; kwargs...)
end

function CommonSolve.solve(
        prob::IntervalNonlinearProblem,
        alg::AbstractBracketingAlgorithm, args...; sensealg = nothing, kwargs...
    )
    return bracketingnonlinear_solve_up(
        prob::IntervalNonlinearProblem, sensealg, prob.p, alg, args...; kwargs...
    )
end

function bracketingnonlinear_solve_up(
        prob::IntervalNonlinearProblem, sensealg, p, alg, args...; kwargs...
    )
    return SciMLBase.__solve(prob, alg, args...; kwargs...)
end

@setup_workload begin
    for T in (Float32, Float64)
        prob_brack = IntervalNonlinearProblem{false}(
            (u, p) -> u^2 - p, T.((0.0, 2.0)), T(2)
        )
        algs = (Alefeld(), Bisection(), Brent(), Falsi(), ITP(), Ridder())

        @compile_workload begin
            @sync for alg in algs
                Threads.@spawn CommonSolve.solve(prob_brack, alg; abstol = 1.0e-6)
            end
        end
    end
end

@reexport using SciMLBase, NonlinearSolveBase

export Alefeld, Bisection, Brent, Falsi, ITP, Muller, Ridder, ModAB

end
