module SimpleNonlinearSolve

import PrecompileTools: @compile_workload, @setup_workload, @recompile_invalidations

@recompile_invalidations begin
    using ADTypes,
        ArrayInterface, ConcreteStructs, DiffEqBase, Reexport, LinearAlgebra,
        SciMLBase

    import DiffEqBase: AbstractNonlinearTerminationMode,
        AbstractSafeNonlinearTerminationMode, AbstractSafeBestNonlinearTerminationMode,
        NonlinearSafeTerminationReturnCode, get_termination_mode
    using FiniteDiff, ForwardDiff
    import ForwardDiff: Dual
    import SciMLBase: AbstractNonlinearAlgorithm, build_solution, isinplace
    import StaticArraysCore: StaticArray, SVector, SMatrix, SArray, MArray
end

@reexport using ADTypes, SciMLBase

# const NNlibExtLoaded = Ref{Bool}(false)

abstract type AbstractSimpleNonlinearSolveAlgorithm <: AbstractNonlinearAlgorithm end
abstract type AbstractBracketingAlgorithm <: AbstractSimpleNonlinearSolveAlgorithm end
abstract type AbstractNewtonAlgorithm <: AbstractSimpleNonlinearSolveAlgorithm end

include("utils.jl")
include("rewrite_inplace.jl")

# Nonlinear Solvera
include("nlsolve/raphson.jl")
include("nlsolve/broyden.jl")
# include("nlsolve/lbroyden.jl")
include("nlsolve/klement.jl")
# include("nlsolve/trustRegion.jl")
# include("nlsolve/halley.jl")
# include("nlsolve/dfsane.jl")

# Interval Nonlinear Solvers
include("bracketing/bisection.jl")
include("bracketing/falsi.jl")
include("bracketing/ridder.jl")
include("bracketing/brent.jl")
include("bracketing/alefeld.jl")
include("bracketing/itp.jl")

# AD
# include("ad.jl")

## Default algorithm

# Set the default bracketing method to ITP

function SciMLBase.solve(prob::IntervalNonlinearProblem; kwargs...)
    return solve(prob, ITP(); kwargs...)
end

function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::Nothing,
        args...; kwargs...)
    return solve(prob, ITP(), args...; kwargs...)
end

# import PrecompileTools

@setup_workload begin
    for T in (Float32, Float64)
        # prob_no_brack = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))
        #         for alg in (SimpleNewtonRaphson, SimpleHalley, Broyden, Klement, SimpleTrustRegion,
        #             SimpleDFSane)
        #             solve(prob_no_brack, alg(), abstol = T(1e-2))
        #         end

        #         #=
        #         for alg in (SimpleNewtonRaphson,)
        #             for u0 in ([1., 1.], StaticArraysCore.SA[1.0, 1.0])
        #                 u0 = T.(.1)
        #                 probN = NonlinearProblem{false}((u,p) -> u .* u .- p, u0, T(2))
        #                 solve(probN, alg(), tol = T(1e-2))
        #             end
        #         end
        #         =#

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

export SimpleBroyden, SimpleGaussNewton, SimpleKlement, SimpleNewtonRaphson
# SimpleDFSane, SimpleTrustRegion, SimpleHalley, LBroyden
export Alefeld, Bisection, Brent, Falsi, ITP, Ridder

end # module
