module NonlinearSolve

using Reexport
using UnPack: @unpack
using FiniteDiff, ForwardDiff
using ForwardDiff: Dual
using LinearAlgebra
using StaticArraysCore
using RecursiveArrayTools
import ArrayInterfaceCore
import LinearSolve
using DiffEqBase
using SparseDiffTools

@reexport using SciMLBase
@reexport using SimpleNonlinearSolve

import SciMLBase: _unwrap_val

abstract type AbstractNonlinearSolveAlgorithm <: SciMLBase.AbstractNonlinearAlgorithm end
abstract type AbstractNewtonAlgorithm{CS, AD, FDT, ST, CJ} <:
              AbstractNonlinearSolveAlgorithm end

function SciMLBase.__solve(prob::NonlinearProblem,
                           alg::AbstractNonlinearSolveAlgorithm, args...;
                           kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    sol = solve!(cache)
end

include("utils.jl")
include("jacobian.jl")
include("raphson.jl")
include("ad.jl")

import SnoopPrecompile

SnoopPrecompile.@precompile_all_calls begin for T in (Float32, Float64)
    prob = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))
    for alg in (NewtonRaphson,)
        solve(prob, alg(), abstol = T(1e-2))
    end

    prob = NonlinearProblem{true}((du, u, p) -> du[1] = u[1] * u[1] - p[1], T[0.1], T[2])
    for alg in (NewtonRaphson,)
        solve(prob, alg(), abstol = T(1e-2))
    end
end end

export NewtonRaphson

end # module
