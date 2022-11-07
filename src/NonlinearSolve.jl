module NonlinearSolve

using Reexport
using UnPack: @unpack
using FiniteDiff, ForwardDiff
using ForwardDiff: Dual
using Setfield
using StaticArrays
using RecursiveArrayTools
using LinearAlgebra
import ArrayInterfaceCore

@reexport using SciMLBase

abstract type AbstractNonlinearSolveAlgorithm <: SciMLBase.AbstractNonlinearAlgorithm end
abstract type AbstractBracketingAlgorithm <: AbstractNonlinearSolveAlgorithm end
abstract type AbstractNewtonAlgorithm{CS, AD, FDT, ST, CJ} <:
              AbstractNonlinearSolveAlgorithm end
abstract type AbstractImmutableNonlinearSolver <: AbstractNonlinearSolveAlgorithm end

using SciMLBase: _unwrap_val
include("utils.jl")
include("jacobian.jl")
include("types.jl")
include("solve.jl")
include("bisection.jl")
include("falsi.jl")
include("raphson.jl")
include("scalar.jl")

import SnoopPrecompile

SnoopPrecompile.@precompile_all_calls begin for T in (Float32, Float64)
    prob_no_brack = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))
    for alg in (NewtonRaphson,)
        solve(prob_no_brack, alg(), tol = T(1e-2))
    end
    #TODO this is broken?
    #for alg in (NewtonRaphson,)
    #    for u0 in ([1., 1.], @SVector[1.0, 1.0])
    #        u0 = T.(.1)
    #        probN = NonlinearProblem{false}((u,p) -> u .* u .- p, u0, T(2))
    #        solve(probN, alg(), tol = T(1e-2))
    #    end
    #end
    prob_brack = NonlinearProblem{false}((u, p) -> u * u - p, T.((0.0, 2.0)), T(2))
    for alg in (Bisection, Falsi)
        solve(prob_brack, alg(), tol = T(1e-2))
    end
end end

# DiffEq styled algorithms
export Bisection, Falsi, NewtonRaphson

end # module
