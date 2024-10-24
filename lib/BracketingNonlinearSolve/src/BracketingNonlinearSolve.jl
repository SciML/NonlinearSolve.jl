module BracketingNonlinearSolve

using ConcreteStructs: @concrete

using CommonSolve: CommonSolve, solve
using NonlinearSolveBase: NonlinearSolveBase
using SciMLBase: SciMLBase, AbstractNonlinearAlgorithm, IntervalNonlinearProblem, ReturnCode

using PrecompileTools: @compile_workload, @setup_workload

abstract type AbstractBracketingAlgorithm <: AbstractNonlinearAlgorithm end

include("common.jl")

include("alefeld.jl")
include("bisection.jl")
include("brent.jl")
include("falsi.jl")
include("itp.jl")
include("ridder.jl")

# Default Algorithm
function CommonSolve.solve(prob::IntervalNonlinearProblem; kwargs...)
    return CommonSolve.solve(prob, ITP(); kwargs...)
end
function CommonSolve.solve(prob::IntervalNonlinearProblem, nothing, args...; kwargs...)
    return CommonSolve.solve(prob, ITP(), args...; kwargs...)
end

@setup_workload begin
    for T in (Float32, Float64)
        prob_brack = IntervalNonlinearProblem{false}(
            (u, p) -> u^2 - p, T.((0.0, 2.0)), T(2))
        algs = (Alefeld(), Bisection(), Brent(), Falsi(), ITP(), Ridder())

        @compile_workload begin
            for alg in algs
                CommonSolve.solve(prob_brack, alg; abstol = 1e-6)
            end
        end
    end
end

export IntervalNonlinearProblem
export solve

export Alefeld, Bisection, Brent, Falsi, ITP, Ridder

end
