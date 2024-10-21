module NonlinearSolve

using Reexport: @reexport
using PrecompileTools: @compile_workload, @setup_workload

using ArrayInterface: ArrayInterface, can_setindex, restructure, fast_scalar_indexing,
                      ismutable
using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase, AbstractNonlinearTerminationMode,
                  AbstractSafeBestNonlinearTerminationMode, AbsNormTerminationMode,
                  AbsSafeBestTerminationMode, AbsSafeTerminationMode, AbsTerminationMode,
                  NormTerminationMode, RelNormTerminationMode, RelSafeBestTerminationMode,
                  RelSafeTerminationMode, RelTerminationMode,
                  SimpleNonlinearSolveTerminationMode, SteadyStateDiffEqTerminationMode
using FastBroadcast: @..
using FastClosures: @closure
using LazyArrays: LazyArrays, ApplyArray, cache
using LinearAlgebra: LinearAlgebra, ColumnNorm, Diagonal, I, LowerTriangular, Symmetric,
                     UpperTriangular, axpy!, cond, diag, diagind, dot, issuccess, istril,
                     istriu, lu, mul!, norm, pinv, tril!, triu!
using LineSearch: LineSearch, AbstractLineSearchAlgorithm, AbstractLineSearchCache,
                  NoLineSearch, RobustNonMonotoneLineSearch, BackTracking, LineSearchesJL
using LinearSolve: LinearSolve, LUFactorization, QRFactorization,
                   needs_concrete_A, AbstractFactorization,
                   DefaultAlgorithmChoice, DefaultLinearSolver
using MaybeInplace: @bb
using Printf: @printf
using Preferences: Preferences, @load_preference, @set_preferences!
using RecursiveArrayTools: recursivecopy!
using SciMLBase: AbstractNonlinearAlgorithm, AbstractNonlinearProblem, _unwrap_val,
                 isinplace, NLStats
using SciMLOperators: AbstractSciMLOperator
using StaticArraysCore: StaticArray, SVector, SArray, MArray, Size, SMatrix
using SymbolicIndexingInterface: SymbolicIndexingInterface, ParameterIndexingProxy,
                                 symbolic_container, parameter_values, state_values, getu,
                                 setu

# AD Support
using ADTypes: ADTypes, AbstractADType, AutoFiniteDiff, AutoForwardDiff,
               AutoPolyesterForwardDiff, AutoZygote, AutoEnzyme, AutoSparse,
               NoSparsityDetector, KnownJacobianSparsityDetector
using DifferentiationInterface: DifferentiationInterface, Constant
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff, Dual
using SciMLJacobianOperators: AbstractJacobianOperator, JacobianOperator, VecJacOperator,
                              JacVecOperator, StatefulJacobianOperator

## Sparse AD Support
using SparseArrays: AbstractSparseMatrix, SparseMatrixCSC
using SparseConnectivityTracer: TracerSparsityDetector # This can be dropped in the next release
using SparseMatrixColorings: ConstantColoringAlgorithm, GreedyColoringAlgorithm,
                             LargestFirst

@reexport using SciMLBase, SimpleNonlinearSolve

const DI = DifferentiationInterface

const True = Val(true)
const False = Val(false)

include("abstract_types.jl")
include("timer_outputs.jl")
include("internal/helpers.jl")

include("descent/common.jl")
include("descent/newton.jl")
include("descent/steepest.jl")
include("descent/dogleg.jl")
include("descent/damped_newton.jl")
include("descent/geodesic_acceleration.jl")

include("internal/jacobian.jl")
include("internal/forward_diff.jl")
include("internal/linear_solve.jl")
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

@setup_workload begin
    nlfuncs = ((NonlinearFunction{false}((u, p) -> u .* u .- p), 0.1),
        (NonlinearFunction{true}((du, u, p) -> du .= u .* u .- p), [0.1]))
    probs_nls = NonlinearProblem[]
    for (fn, u0) in nlfuncs
        push!(probs_nls, NonlinearProblem(fn, u0, 2.0))
    end

    nls_algs = (
        NewtonRaphson(),
        TrustRegion(),
        LevenbergMarquardt(),
        # PseudoTransient(),
        Broyden(),
        Klement(),
        # DFSane(),
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
        # LevenbergMarquardt(; linsolve = LUFactorization()),
        # GaussNewton(; linsolve = LUFactorization()),
        # TrustRegion(; linsolve = LUFactorization()),
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

# Core Algorithms
export NewtonRaphson, PseudoTransient, Klement, Broyden, LimitedMemoryBroyden, DFSane
export GaussNewton, LevenbergMarquardt, TrustRegion
export NonlinearSolvePolyAlgorithm, RobustMultiNewton, FastShortcutNonlinearPolyalg,
       FastShortcutNLLSPolyalg

# Extension Algorithms
export LeastSquaresOptimJL, FastLevenbergMarquardtJL, CMINPACK, NLsolveJL, NLSolversJL,
       FixedPointAccelerationJL, SpeedMappingJL, SIAMFANLEquationsJL

# Advanced Algorithms -- Without Bells and Whistles
export GeneralizedFirstOrderAlgorithm, ApproximateJacobianSolveAlgorithm, GeneralizedDFSane

# Descent Algorithms
export NewtonDescent, SteepestDescent, Dogleg, DampedNewtonDescent, GeodesicAcceleration

# Globalization
## Line Search Algorithms
export LineSearch, BackTracking, NoLineSearch, RobustNonMonotoneLineSearch, LineSearchesJL
## Trust Region Algorithms
export RadiusUpdateSchemes

# Export the termination conditions from NonlinearSolveBase
export SteadyStateDiffEqTerminationMode, SimpleNonlinearSolveTerminationMode,
       NormTerminationMode, RelTerminationMode, RelNormTerminationMode, AbsTerminationMode,
       AbsNormTerminationMode, RelSafeTerminationMode, AbsSafeTerminationMode,
       RelSafeBestTerminationMode, AbsSafeBestTerminationMode

# Tracing Functionality
export TraceAll, TraceMinimal, TraceWithJacobianConditionNumber

# Reexport ADTypes
export AutoFiniteDiff, AutoForwardDiff, AutoPolyesterForwardDiff, AutoZygote, AutoEnzyme,
       AutoSparse

end
