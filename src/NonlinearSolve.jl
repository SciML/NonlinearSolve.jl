module NonlinearSolve

using Reexport: @reexport
using PrecompileTools: @compile_workload, @setup_workload

using ArrayInterface: ArrayInterface, can_setindex, restructure, fast_scalar_indexing,
                      ismutable
using CommonSolve: solve, init, solve!
using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase # Needed for `init` / `solve` dispatches
using FastClosures: @closure
using LinearAlgebra: LinearAlgebra, Diagonal, I, LowerTriangular, Symmetric,
                     UpperTriangular, axpy!, cond, diag, diagind, dot, issuccess, istril,
                     istriu, lu, mul!, norm, pinv, tril!, triu!
using LineSearch: LineSearch, AbstractLineSearchCache, LineSearchesJL, NoLineSearch,
                  RobustNonMonotoneLineSearch, BackTracking, LiFukushimaLineSearch
using LinearSolve: LinearSolve
using MaybeInplace: @bb
using NonlinearSolveBase: NonlinearSolveBase,
                          nonlinearsolve_forwarddiff_solve, nonlinearsolve_dual_solution,
                          nonlinearsolve_∂f_∂p, nonlinearsolve_∂f_∂u,
                          L2_NORM,
                          AbsNormTerminationMode, AbstractNonlinearTerminationMode,
                          AbstractSafeBestNonlinearTerminationMode,
                          select_forward_mode_autodiff, select_reverse_mode_autodiff,
                          select_jacobian_autodiff,
                          construct_jacobian_cache,
                          DescentResult,
                          SteepestDescent, NewtonDescent, DampedNewtonDescent, Dogleg,
                          GeodesicAcceleration,
                          reset_timer!, @static_timeit,
                          init_nonlinearsolve_trace, update_trace!, reset!

using NonlinearSolveQuasiNewton: Broyden, Klement

using Preferences: Preferences, set_preferences!
using RecursiveArrayTools: recursivecopy!
using SciMLBase: SciMLBase, AbstractNonlinearAlgorithm, AbstractNonlinearProblem,
                 _unwrap_val, isinplace, NLStats, NonlinearFunction,
                 NonlinearLeastSquaresProblem, NonlinearProblem, ReturnCode, get_du, step!,
                 set_u!, LinearProblem, IdentityOperator
using SciMLOperators: AbstractSciMLOperator
using SimpleNonlinearSolve: SimpleNonlinearSolve
using StaticArraysCore: StaticArray, SVector, SArray, MArray, Size, SMatrix

# AD Support
using ADTypes: ADTypes, AbstractADType, AutoFiniteDiff, AutoForwardDiff,
               AutoPolyesterForwardDiff, AutoZygote, AutoEnzyme, AutoSparse
using DifferentiationInterface: DifferentiationInterface
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff, Dual
using SciMLJacobianOperators: VecJacOperator, JacVecOperator, StatefulJacobianOperator

## Sparse AD Support
using SparseArrays: AbstractSparseMatrix, SparseMatrixCSC
using SparseMatrixColorings: SparseMatrixColorings # NOTE: This triggers an extension in NonlinearSolveBase

const DI = DifferentiationInterface

include("timer_outputs.jl")
include("internal/helpers.jl")

include("algorithms/extension_algs.jl")

include("utils.jl")
include("default.jl")

const ALL_SOLVER_TYPES = [
    Nothing, AbstractNonlinearSolveAlgorithm, GeneralizedDFSane,
    GeneralizedFirstOrderAlgorithm, ApproximateJacobianSolveAlgorithm,
    LeastSquaresOptimJL, FastLevenbergMarquardtJL, NLsolveJL, NLSolversJL,
    SpeedMappingJL, FixedPointAccelerationJL, SIAMFANLEquationsJL,
    CMINPACK, PETScSNES,
    NonlinearSolvePolyAlgorithm{:NLLS, <:Any}, NonlinearSolvePolyAlgorithm{:NLS, <:Any}
]

include("internal/forward_diff.jl") # we need to define after the algorithms

@setup_workload begin
    nlfuncs = (
        (NonlinearFunction{false}((u, p) -> u .* u .- p), 0.1),
        (NonlinearFunction{true}((du, u, p) -> du .= u .* u .- p), [0.1])
    )
    probs_nls = NonlinearProblem[]
    for (fn, u0) in nlfuncs
        push!(probs_nls, NonlinearProblem(fn, u0, 2.0))
    end

    nls_algs = (
        nothing
    )

    probs_nlls = NonlinearLeastSquaresProblem[]
    nlfuncs = (
        (NonlinearFunction{false}((u, p) -> (u .^ 2 .- p)[1:1]), [0.1, 0.0]),
        (NonlinearFunction{false}((u, p) -> vcat(u .* u .- p, u .* u .- p)), [0.1, 0.1]),
        (
            NonlinearFunction{true}(
                (du, u, p) -> du[1] = u[1] * u[1] - p, resid_prototype = zeros(1)),
            [0.1, 0.0]),
        (
            NonlinearFunction{true}((du, u, p) -> du .= vcat(u .* u .- p, u .* u .- p),
                resid_prototype = zeros(4)),
            [0.1, 0.1]
        )
    )
    for (fn, u0) in nlfuncs
        push!(probs_nlls, NonlinearLeastSquaresProblem(fn, u0, 2.0))
    end

    nlls_algs = (
        LevenbergMarquardt(),
        GaussNewton(),
        TrustRegion(),
        nothing
    )

    @compile_workload begin
        for prob in probs_nls, alg in nls_algs
            solve(prob, alg; abstol = 1e-2, verbose = false)
        end
        for prob in probs_nlls, alg in nlls_algs
            solve(prob, alg; abstol = 1e-2, verbose = false)
        end
    end
end

# Rexexports
@reexport using SciMLBase, NonlinearSolveBase
@reexport using NonlinearSolveFirstOrder, NonlinearSolveSpectralMethods,
                NonlinearSolveQuasiNewton, SimpleNonlinearSolve
@reexport using LineSearch
@reexport using ADTypes

export NonlinearSolvePolyAlgorithm, RobustMultiNewton,
       FastShortcutNonlinearPolyalg, FastShortcutNLLSPolyalg

# Extension Algorithms
export LeastSquaresOptimJL, FastLevenbergMarquardtJL, NLsolveJL, NLSolversJL,
       FixedPointAccelerationJL, SpeedMappingJL, SIAMFANLEquationsJL
export PETScSNES, CMINPACK

end
