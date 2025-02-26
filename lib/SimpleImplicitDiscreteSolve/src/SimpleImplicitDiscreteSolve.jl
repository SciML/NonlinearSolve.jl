module ImplicitDiscreteSolve

using SciMLBase: AbstractNonlinearProblem
using SciMLBase
using NonlinearSolveBase
using SimpleNonlinearSolve
using SymbolicIndexingInterface
using LinearAlgebra
using ADTypes
using UnPack
import OrdinaryDiffEqCore: OrdinaryDiffEqAlgorithm, alg_cache, OrdinaryDiffEqMutableCache, OrdinaryDiffEqConstantCache, get_fsalfirstlast, isfsal, initialize!, perform_step!, isdiscretecache, isdiscretealg, alg_order, beta2_default, beta1_default, dt_required
import CommonSolve
import DifferentiationInterface as DI

using Reexport
@reexport using DiffEqBase

"""
    IDSolve(alg; autodiff = true, kwargs...)

Solver for `ImplicitDiscreteSystems`. `alg` is the NonlinearSolve algorithm that is used to solve for the next timestep at each step.
"""
struct SimpleIDSolve{algType} <: OrdinaryDiffEqAlgorithm end

include("cache.jl")
include("solve.jl")
include("alg_utils.jl")

export SimpleIDSolve

end
