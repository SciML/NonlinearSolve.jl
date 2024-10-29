module NonlinearSolve

using Reexport: @reexport
using PrecompileTools: @compile_workload, @setup_workload

using ArrayInterface: ArrayInterface
using CommonSolve: CommonSolve, solve, solve!
using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase # Needed for `init` / `solve` dispatches
using FastClosures: @closure
using LinearAlgebra: LinearAlgebra, norm
using LineSearch: BackTracking
using NonlinearSolveBase: NonlinearSolveBase, InternalAPI, AbstractNonlinearSolveAlgorithm,
                          AbstractNonlinearSolveCache, Utils, L2_NORM

using Preferences: set_preferences!
using SciMLBase: SciMLBase, NLStats, ReturnCode, AbstractNonlinearProblem, NonlinearProblem,
                 NonlinearLeastSquaresProblem
using SymbolicIndexingInterface: SymbolicIndexingInterface
using StaticArraysCore: StaticArray

# Default Algorithm
using NonlinearSolveFirstOrder: NewtonRaphson, TrustRegion, LevenbergMarquardt, GaussNewton,
                                RUS
using NonlinearSolveQuasiNewton: Broyden, Klement
using SimpleNonlinearSolve: SimpleBroyden, SimpleKlement

# Default AD Support
using FiniteDiff: FiniteDiff    # Default Finite Difference Method
using ForwardDiff: ForwardDiff  # Default Forward Mode AD

# Sparse AD Support: Implemented via extensions
using SparseArrays: SparseArrays
using SparseMatrixColorings: SparseMatrixColorings

const SII = SymbolicIndexingInterface

include("helpers.jl")

include("polyalg.jl")
# include("extension_algs.jl")

include("default.jl")

# const ALL_SOLVER_TYPES = [
#     Nothing, AbstractNonlinearSolveAlgorithm, GeneralizedDFSane,
#     GeneralizedFirstOrderAlgorithm, ApproximateJacobianSolveAlgorithm,
#     LeastSquaresOptimJL, FastLevenbergMarquardtJL, NLsolveJL, NLSolversJL,
#     SpeedMappingJL, FixedPointAccelerationJL, SIAMFANLEquationsJL,
#     CMINPACK, PETScSNES,
#     NonlinearSolvePolyAlgorithm{:NLLS, <:Any}, NonlinearSolvePolyAlgorithm{:NLS, <:Any}
# ]

# include("internal/forward_diff.jl") # we need to define after the algorithms

@setup_workload begin
    include(joinpath(@__DIR__, "..", "common", "nonlinear_problem_workloads.jl"))
    include(joinpath(@__DIR__, "..", "common", "nlls_problem_workloads.jl"))

    @compile_workload begin
        @sync begin
            for prob in nonlinear_problems
                Threads.@spawn CommonSolve.solve(
                    prob, nothing; abstol = 1e-2, verbose = false
                )
            end

            for prob in nlls_problems
                Threads.@spawn CommonSolve.solve(
                    prob, nothing; abstol = 1e-2, verbose = false
                )
            end
        end
    end
end

# Rexexports
@reexport using SciMLBase, NonlinearSolveBase, LineSearch, ADTypes
@reexport using NonlinearSolveFirstOrder, NonlinearSolveSpectralMethods,
                NonlinearSolveQuasiNewton, SimpleNonlinearSolve, BracketingNonlinearSolve
@reexport using LinearSolve

# Poly Algorithms
export NonlinearSolvePolyAlgorithm,
       RobustMultiNewton, FastShortcutNonlinearPolyalg, FastShortcutNLLSPolyalg

# Extension Algorithms
export LeastSquaresOptimJL, FastLevenbergMarquardtJL, NLsolveJL, NLSolversJL,
       FixedPointAccelerationJL, SpeedMappingJL, SIAMFANLEquationsJL
export PETScSNES, CMINPACK

end
