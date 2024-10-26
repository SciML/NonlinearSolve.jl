module NonlinearSolve

using Reexport: @reexport
using PrecompileTools: @compile_workload, @setup_workload

using ArrayInterface: ArrayInterface, can_setindex, restructure, fast_scalar_indexing,
                      ismutable
using CommonSolve: solve, init, solve!
using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase # Needed for `init` / `solve` dispatches
using FastClosures: @closure
using LazyArrays: LazyArrays, ApplyArray, cache
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
                          reset_timer!, @static_timeit

# XXX: Remove
import NonlinearSolveBase: InternalAPI, concrete_jac, supports_line_search,
                           supports_trust_region, last_step_accepted, get_linear_solver,
                           AbstractDampingFunction, AbstractDampingFunctionCache,
                           requires_normal_form_jacobian, requires_normal_form_rhs,
                           returns_norm_form_damping, get_timer_output

using Printf: @printf
using Preferences: Preferences, set_preferences!
using RecursiveArrayTools: recursivecopy!
using SciMLBase: SciMLBase, AbstractNonlinearAlgorithm, AbstractNonlinearProblem,
                 _unwrap_val, isinplace, NLStats, NonlinearFunction,
                 NonlinearLeastSquaresProblem, NonlinearProblem, ReturnCode, get_du, step!,
                 set_u!, LinearProblem, IdentityOperator
using SciMLOperators: AbstractSciMLOperator
using SimpleNonlinearSolve: SimpleNonlinearSolve
using StaticArraysCore: StaticArray, SVector, SArray, MArray, Size, SMatrix
using SymbolicIndexingInterface: SymbolicIndexingInterface, ParameterIndexingProxy,
                                 symbolic_container, parameter_values, state_values, getu,
                                 setu

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

const True = Val(true)
const False = Val(false)

include("abstract_types.jl")
include("timer_outputs.jl")
include("internal/helpers.jl")

include("internal/termination.jl")
include("internal/tracing.jl")
include("internal/approximate_initialization.jl")

include("globalization/line_search.jl")
include("globalization/trust_region.jl")

include("core/generic.jl")
include("core/approximate_jacobian.jl")
include("core/generalized_first_order.jl")
include("core/spectral_methods.jl")
include("core/noinit.jl")

include("algorithms/raphson.jl")
include("algorithms/pseudo_transient.jl")
include("algorithms/broyden.jl")
include("algorithms/klement.jl")
include("algorithms/lbroyden.jl")
include("algorithms/dfsane.jl")
include("algorithms/gauss_newton.jl")
include("algorithms/levenberg_marquardt.jl")
include("algorithms/trust_region.jl")
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
        NewtonRaphson(),
        TrustRegion(),
        LevenbergMarquardt(),
        Broyden(),
        Klement(),
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
        @sync begin
            for T in (Float32, Float64), (fn, u0) in nlfuncs
                Threads.@spawn NonlinearProblem(fn, T.(u0), T(2))
            end
            for (fn, u0) in nlfuncs
                Threads.@spawn NonlinearLeastSquaresProblem(fn, u0, 2.0)
            end
            for prob in probs_nls, alg in nls_algs
                Threads.@spawn solve(prob, alg; abstol = 1e-2, verbose = false)
            end
            for prob in probs_nlls, alg in nlls_algs
                Threads.@spawn solve(prob, alg; abstol = 1e-2, verbose = false)
            end
        end
    end
end

# Rexexports
@reexport using SciMLBase, SimpleNonlinearSolve, NonlinearSolveBase

# Core Algorithms
export NewtonRaphson, PseudoTransient, Klement, Broyden, LimitedMemoryBroyden, DFSane
export GaussNewton, LevenbergMarquardt, TrustRegion
export NonlinearSolvePolyAlgorithm, RobustMultiNewton, FastShortcutNonlinearPolyalg,
       FastShortcutNLLSPolyalg

# Extension Algorithms
export LeastSquaresOptimJL, FastLevenbergMarquardtJL, NLsolveJL, NLSolversJL,
       FixedPointAccelerationJL, SpeedMappingJL, SIAMFANLEquationsJL
export PETScSNES, CMINPACK

# Advanced Algorithms -- Without Bells and Whistles
export GeneralizedFirstOrderAlgorithm, ApproximateJacobianSolveAlgorithm, GeneralizedDFSane

# Globalization
## Line Search Algorithms
export LineSearch, BackTracking, NoLineSearch, RobustNonMonotoneLineSearch,
       LiFukushimaLineSearch, LineSearchesJL
## Trust Region Algorithms
export RadiusUpdateSchemes

# Tracing Functionality
export TraceAll, TraceMinimal, TraceWithJacobianConditionNumber

# Reexport ADTypes
export AutoFiniteDiff, AutoForwardDiff, AutoPolyesterForwardDiff, AutoZygote, AutoEnzyme,
       AutoSparse

end
