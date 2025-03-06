module SimpleImplicitDiscreteSolve

using SciMLBase
using SimpleNonlinearSolve
using UnPack
import OrdinaryDiffEqCore: OrdinaryDiffEqAlgorithm, alg_cache, OrdinaryDiffEqMutableCache, OrdinaryDiffEqConstantCache, get_fsalfirstlast, isfsal, initialize!, perform_step!, isdiscretecache, isdiscretealg, alg_order, beta2_default, beta1_default, dt_required

using Reexport
@reexport using DiffEqBase

"""
    SimpleIDSolve()

Simple solver for `ImplicitDiscreteSystems`. Uses `SimpleNewtonRaphson` to solve for the next state at every timestep.
"""
struct SimpleIDSolve <: OrdinaryDiffEqAlgorithm end

include("cache.jl")
include("solve.jl")
include("alg_utils.jl")

export SimpleIDSolve

end
