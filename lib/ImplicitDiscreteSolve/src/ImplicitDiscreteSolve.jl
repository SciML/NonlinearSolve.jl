module ImplicitDiscreteSolve

using SciMLBase: AbstractNonlinearProblem
using SciMLBase
using NonlinearSolveBase
using SymbolicIndexingInterface
using LinearAlgebra
using ADTypes
using UnPack
import OrdinaryDiffEqCore: OrdinaryDiffEqAlgorithm, alg_cache, OrdinaryDiffEqMutableCache, OrdinaryDiffEqConstantCache, get_fsalfirstlast, initialize!, perform_step!
import CommonSolve
import DifferentiationInterface as DI

using Reexport
@reexport using DiffEqBase

"""
    IDSolve(alg; autodiff = true, kwargs...)

Solver for `ImplicitDiscreteSystems`. `alg` is the NonlinearSolve algorithm that is used to solve for the next timestep at each step.
"""
struct IDSolve{algType} <: OrdinaryDiffEqAlgorithm
    nlsolve::algType
end

include("cache.jl")
include("solve.jl")

export IDSolve

end
