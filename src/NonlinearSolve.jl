module NonlinearSolve

if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end

import Reexport: @reexport
import PrecompileTools: @recompile_invalidations, @compile_workload, @setup_workload

@recompile_invalidations begin
    using Accessors, ADTypes, ConcreteStructs, DiffEqBase, FastBroadcast, FastClosures,
          LazyArrays, LineSearches, LinearAlgebra, LinearSolve, MaybeInplace, Preferences,
          Printf, SciMLBase, SimpleNonlinearSolve, SparseArrays, SparseDiffTools

    import ArrayInterface: undefmatrix, can_setindex, restructure, fast_scalar_indexing
    import DiffEqBase: AbstractNonlinearTerminationMode,
                       AbstractSafeNonlinearTerminationMode,
                       AbstractSafeBestNonlinearTerminationMode,
                       NonlinearSafeTerminationReturnCode, get_termination_mode
    import FiniteDiff
    import ForwardDiff
    import ForwardDiff: Dual
    import LinearSolve: ComposePreconditioner, InvPreconditioner, needs_concrete_A
    import RecursiveArrayTools: recursivecopy!, recursivefill!

    import SciMLBase: AbstractNonlinearAlgorithm, JacobianWrapper, AbstractNonlinearProblem,
                      AbstractSciMLOperator, NLStats, _unwrap_val, has_jac, isinplace
    import SparseDiffTools: AbstractSparsityDetection, AutoSparseEnzyme
    import StaticArraysCore: StaticArray, SVector, SArray, MArray, Size, SMatrix, MMatrix
end

@reexport using ADTypes, LineSearches, SciMLBase, SimpleNonlinearSolve

const AbstractSparseADType = Union{ADTypes.AbstractSparseFiniteDifferences,
    ADTypes.AbstractSparseForwardMode, ADTypes.AbstractSparseReverseMode}

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
include("descent/multistep.jl")

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

include("algorithms/raphson.jl")
include("algorithms/pseudo_transient.jl")
include("algorithms/multistep.jl")
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
        (NonlinearFunction{false}((u, p) -> u .* u .- p), [0.1]),
        (NonlinearFunction{true}((du, u, p) -> du .= u .* u .- p), [0.1]))
    probs_nls = NonlinearProblem[]
    for T in (Float32, Float64), (fn, u0) in nlfuncs
        push!(probs_nls, NonlinearProblem(fn, T.(u0), T(2)))
    end

    nls_algs = (NewtonRaphson(), TrustRegion(), LevenbergMarquardt(), PseudoTransient(),
        Broyden(), Klement(), DFSane(), nothing)

    probs_nlls = NonlinearLeastSquaresProblem[]
    nlfuncs = ((NonlinearFunction{false}((u, p) -> (u .^ 2 .- p)[1:1]), [0.1, 0.0]),
        (NonlinearFunction{false}((u, p) -> vcat(u .* u .- p, u .* u .- p)), [0.1, 0.1]),
        (
            NonlinearFunction{true}((du, u, p) -> du[1] = u[1] * u[1] - p,
                resid_prototype = zeros(1)),
            [0.1, 0.0]),
        (
            NonlinearFunction{true}((du, u, p) -> du .= vcat(u .* u .- p, u .* u .- p),
                resid_prototype = zeros(4)),
            [0.1, 0.1]))
    for (fn, u0) in nlfuncs
        push!(probs_nlls, NonlinearLeastSquaresProblem(fn, u0, 2.0))
    end
    nlfuncs = ((NonlinearFunction{false}((u, p) -> (u .^ 2 .- p)[1:1]), Float32[0.1, 0.0]),
        (NonlinearFunction{false}((u, p) -> vcat(u .* u .- p, u .* u .- p)),
            Float32[0.1, 0.1]),
        (
            NonlinearFunction{true}((du, u, p) -> du[1] = u[1] * u[1] - p,
                resid_prototype = zeros(Float32, 1)),
            Float32[0.1, 0.0]),
        (
            NonlinearFunction{true}((du, u, p) -> du .= vcat(u .* u .- p, u .* u .- p),
                resid_prototype = zeros(Float32, 4)),
            Float32[0.1, 0.1]))
    for (fn, u0) in nlfuncs
        push!(probs_nlls, NonlinearLeastSquaresProblem(fn, u0, 2.0f0))
    end

    nlls_algs = (LevenbergMarquardt(), GaussNewton(), TrustRegion(),
        LevenbergMarquardt(; linsolve = LUFactorization()),
        GaussNewton(; linsolve = LUFactorization()),
        TrustRegion(; linsolve = LUFactorization()), nothing)

    @compile_workload begin
        for prob in probs_nls, alg in nls_algs
            solve(prob, alg; abstol = 1e-2)
        end
        for prob in probs_nlls, alg in nlls_algs
            solve(prob, alg; abstol = 1e-2)
        end
    end
end

# Core Algorithms
export NewtonRaphson, PseudoTransient, Klement, Broyden, LimitedMemoryBroyden, DFSane,
       MultiStepNonlinearSolver
export GaussNewton, LevenbergMarquardt, TrustRegion
export NonlinearSolvePolyAlgorithm,
       RobustMultiNewton, FastShortcutNonlinearPolyalg, FastShortcutNLLSPolyalg

# Extension Algorithms
export LeastSquaresOptimJL, FastLevenbergMarquardtJL, CMINPACK, NLsolveJL,
       FixedPointAccelerationJL, SpeedMappingJL, SIAMFANLEquationsJL

# Advanced Algorithms -- Without Bells and Whistles
export GeneralizedFirstOrderAlgorithm, ApproximateJacobianSolveAlgorithm, GeneralizedDFSane

# Descent Algorithms
export NewtonDescent, SteepestDescent, Dogleg, DampedNewtonDescent,
       GeodesicAcceleration, GenericMultiStepDescent
## Multistep Algorithms
export MultiStepSchemes

# Globalization
## Line Search Algorithms
export LineSearchesJL, NoLineSearch, RobustNonMonotoneLineSearch, LiFukushimaLineSearch
## Trust Region Algorithms
export RadiusUpdateSchemes

# Export the termination conditions from DiffEqBase
export SteadyStateDiffEqTerminationMode, SimpleNonlinearSolveTerminationMode,
       NormTerminationMode, RelTerminationMode, RelNormTerminationMode, AbsTerminationMode,
       AbsNormTerminationMode, RelSafeTerminationMode, AbsSafeTerminationMode,
       RelSafeBestTerminationMode, AbsSafeBestTerminationMode

# Tracing Functionality
export TraceAll, TraceMinimal, TraceWithJacobianConditionNumber

end # module
