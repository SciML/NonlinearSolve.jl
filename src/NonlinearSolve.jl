module NonlinearSolve

if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end

using Reexport: @reexport
using PrecompileTools: @recompile_invalidations, @compile_workload, @setup_workload

@recompile_invalidations begin
    using ADTypes: AutoFiniteDiff, AutoForwardDiff, AutoPolyesterForwardDiff, AutoZygote,
                   AutoEnzyme, AutoSparse
    # FIXME: deprecated, remove in future
    using ADTypes: AutoSparseFiniteDiff, AutoSparseForwardDiff,
                   AutoSparsePolyesterForwardDiff, AutoSparseZygote

    using ArrayInterface: ArrayInterface, can_setindex, restructure, fast_scalar_indexing,
                          ismutable
    using ConcreteStructs: @concrete
    using DiffEqBase: DiffEqBase, AbstractNonlinearTerminationMode,
                      AbstractSafeBestNonlinearTerminationMode, AbsNormTerminationMode,
                      AbsSafeBestTerminationMode, AbsSafeTerminationMode,
                      AbsTerminationMode, NormTerminationMode, RelNormTerminationMode,
                      RelSafeBestTerminationMode, RelSafeTerminationMode,
                      RelTerminationMode, SimpleNonlinearSolveTerminationMode,
                      SteadyStateDiffEqTerminationMode
    using FastBroadcast: @..
    using FastClosures: @closure
    using FiniteDiff: FiniteDiff
    using ForwardDiff: ForwardDiff, Dual
    using LazyArrays: LazyArrays, ApplyArray, cache
    using LinearAlgebra: LinearAlgebra, ColumnNorm, Diagonal, I, LowerTriangular, Symmetric,
                         UpperTriangular, axpy!, cond, diag, diagind, dot, issuccess,
                         istril, istriu, lu, mul!, norm, pinv, tril!, triu!
    using LineSearches: LineSearches
    using LinearSolve: LinearSolve, LUFactorization, QRFactorization, ComposePreconditioner,
                       InvPreconditioner, needs_concrete_A, AbstractFactorization,
                       DefaultAlgorithmChoice, DefaultLinearSolver
    using MaybeInplace: @bb
    using Printf: @printf
    using Preferences: Preferences, @load_preference, @set_preferences!
    using RecursiveArrayTools: recursivecopy!, recursivefill!
    using SciMLBase: AbstractNonlinearAlgorithm, JacobianWrapper, AbstractNonlinearProblem,
                     AbstractSciMLOperator, _unwrap_val, has_jac, isinplace, NLStats
    using SparseArrays: AbstractSparseMatrix, SparseMatrixCSC
    using SparseDiffTools: SparseDiffTools, AbstractSparsityDetection,
                           ApproximateJacobianSparsity, JacPrototypeSparsityDetection,
                           NoSparsityDetection, PrecomputedJacobianColorvec,
                           SymbolicsSparsityDetection, auto_jacvec, auto_jacvec!,
                           auto_vecjac, init_jacobian, num_jacvec, num_jacvec!, num_vecjac,
                           num_vecjac!, sparse_jacobian, sparse_jacobian!,
                           sparse_jacobian_cache
    using StaticArraysCore: StaticArray, SVector, SArray, MArray, Size, SMatrix
    using SymbolicIndexingInterface: SymbolicIndexingInterface, ParameterIndexingProxy,
                                     symbolic_container, parameter_values, state_values,
                                     getu
end

@reexport using SciMLBase, SimpleNonlinearSolve

# Type-Inference Friendly Check for Extension Loading
is_extension_loaded(::Val) = false

const True = Val(true)
const False = Val(false)

include("abstract_types.jl")
include("adtypes.jl")
include("timer_outputs.jl")
include("internal/helpers.jl")

include("descent/common.jl")
include("descent/newton.jl")
include("descent/steepest.jl")
include("descent/dogleg.jl")
include("descent/damped_newton.jl")
include("descent/geodesic_acceleration.jl")

include("internal/operators.jl")
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

    nls_algs = (NewtonRaphson(), TrustRegion(), LevenbergMarquardt(),
        PseudoTransient(), Broyden(), Klement(), DFSane(), nothing)

    probs_nlls = NonlinearLeastSquaresProblem[]
    nlfuncs = ((NonlinearFunction{false}((u, p) -> (u .^ 2 .- p)[1:1]), [0.1, 0.0]),
        (NonlinearFunction{false}((u, p) -> vcat(u .* u .- p, u .* u .- p)), [0.1, 0.1]),
        (
            NonlinearFunction{true}(
                (du, u, p) -> du[1] = u[1] * u[1] - p, resid_prototype = zeros(1)),
            [0.1, 0.0]),
        (
            NonlinearFunction{true}((du, u, p) -> du .= vcat(u .* u .- p, u .* u .- p),
                resid_prototype = zeros(4)),
            [0.1, 0.1]))
    for (fn, u0) in nlfuncs
        push!(probs_nlls, NonlinearLeastSquaresProblem(fn, u0, 2.0))
    end

    nlls_algs = (LevenbergMarquardt(), GaussNewton(), TrustRegion(),
        LevenbergMarquardt(; linsolve = LUFactorization()),
        GaussNewton(; linsolve = LUFactorization()),
        TrustRegion(; linsolve = LUFactorization()), nothing)

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
export LineSearchesJL, NoLineSearch, RobustNonMonotoneLineSearch, LiFukushimaLineSearch
export Static, HagerZhang, MoreThuente, StrongWolfe, BackTracking
## Trust Region Algorithms
export RadiusUpdateSchemes

# Export the termination conditions from DiffEqBase
export SteadyStateDiffEqTerminationMode, SimpleNonlinearSolveTerminationMode,
       NormTerminationMode, RelTerminationMode, RelNormTerminationMode, AbsTerminationMode,
       AbsNormTerminationMode, RelSafeTerminationMode, AbsSafeTerminationMode,
       RelSafeBestTerminationMode, AbsSafeBestTerminationMode

# Tracing Functionality
export TraceAll, TraceMinimal, TraceWithJacobianConditionNumber

# Reexport ADTypes
export AutoFiniteDiff, AutoForwardDiff, AutoPolyesterForwardDiff, AutoZygote, AutoEnzyme,
       AutoSparse
# FIXME: deprecated, remove in future
export AutoSparseFiniteDiff, AutoSparseForwardDiff, AutoSparsePolyesterForwardDiff,
       AutoSparseZygote

end # module
