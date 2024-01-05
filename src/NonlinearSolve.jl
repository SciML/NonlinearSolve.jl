module NonlinearSolve

if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end

import Reexport: @reexport
import PrecompileTools: @recompile_invalidations, @compile_workload, @setup_workload

@recompile_invalidations begin
    using ADTypes, ConcreteStructs, DiffEqBase, FastBroadcast, FastClosures, LazyArrays,
        LineSearches, LinearAlgebra, LinearSolve, MaybeInplace, Printf, SciMLBase,
        SimpleNonlinearSolve, SparseArrays, SparseDiffTools, SumTypes, TimerOutputs

    import ArrayInterface: undefmatrix, can_setindex, restructure, fast_scalar_indexing
    import DiffEqBase: AbstractNonlinearTerminationMode,
        AbstractSafeNonlinearTerminationMode, AbstractSafeBestNonlinearTerminationMode,
        NonlinearSafeTerminationReturnCode, get_termination_mode
    import FiniteDiff
    import ForwardDiff
    import ForwardDiff: Dual
    import LinearSolve: ComposePreconditioner, InvPreconditioner, needs_concrete_A
    import RecursiveArrayTools: recursivecopy!, recursivefill!

    import SciMLBase: AbstractNonlinearAlgorithm, JacobianWrapper, AbstractNonlinearProblem,
        AbstractSciMLOperator, NLStats, _unwrap_val, has_jac, isinplace
    import SparseDiffTools: AbstractSparsityDetection
    import StaticArraysCore: StaticArray, SVector, SArray, MArray, Size, SMatrix, MMatrix
end

@reexport using ADTypes, LineSearches, SciMLBase, SimpleNonlinearSolve

const AbstractSparseADType = Union{ADTypes.AbstractSparseFiniteDifferences,
    ADTypes.AbstractSparseForwardMode, ADTypes.AbstractSparseReverseMode}

# Type-Inference Friendly Check for Extension Loading
is_extension_loaded(::Val) = false

const True = Val(true)
const False = Val(false)

# function SciMLBase.reinit!(cache::AbstractNonlinearSolveCache{iip}, u0 = get_u(cache);
#         p = cache.p, abstol = cache.abstol, reltol = cache.reltol,
#         maxiters = cache.maxiters, alias_u0 = false, termination_condition = missing,
#         kwargs...) where {iip}
#     cache.p = p
#     if iip
#         recursivecopy!(get_u(cache), u0)
#         cache.f(get_fu(cache), get_u(cache), p)
#     else
#         cache.u = __maybe_unaliased(u0, alias_u0)
#         set_fu!(cache, cache.f(cache.u, p))
#     end

#     reset!(cache.trace)

#     # Some algorithms store multiple termination caches
#     if hasfield(typeof(cache), :tc_cache)
#         # TODO: We need an efficient way to reset this upstream
#         tc = termination_condition === missing ? get_termination_mode(cache.tc_cache) :
#              termination_condition
#         abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, get_fu(cache),
#             get_u(cache), tc)
#         cache.tc_cache = tc_cache
#     end

#     if hasfield(typeof(cache), :ls_cache)
#         # TODO: A more efficient way to do this
#         cache.ls_cache = init_linesearch_cache(cache.alg.linesearch, cache.f,
#             get_u(cache), p, get_fu(cache), Val(iip))
#     end

#     hasfield(typeof(cache), :uf) && cache.uf !== nothing && (cache.uf.p = p)

#     cache.abstol = abstol
#     cache.reltol = reltol
#     cache.maxiters = maxiters
#     cache.stats.nf = 1
#     cache.stats.nsteps = 1
#     cache.force_stop = false
#     cache.retcode = ReturnCode.Default

#     __reinit_internal!(cache; u0, p, abstol, reltol, maxiters, alias_u0,
#         termination_condition, kwargs...)

#     return cache
# end

# __reinit_internal!(::AbstractNonlinearSolveCache; kwargs...) = nothing

include("abstract_types.jl")
include("internal/helpers.jl")

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
include("internal/approx_initialization.jl")

include("globalization/line_search.jl")
include("globalization/trust_region.jl")

include("core/generic.jl")
include("core/approximate_jacobian.jl")
include("core/generalized_first_order.jl")
include("core/spectral_methods.jl")

include("algorithms/raphson.jl")
include("algorithms/pseudo_transient.jl")
include("algorithms/broyden.jl")
include("algorithms/klement.jl")
include("algorithms/lbroyden.jl")
include("algorithms/dfsane.jl")
include("algorithms/gradient_descent.jl")
include("algorithms/gauss_newton.jl")
include("algorithms/levenberg_marquardt.jl")
include("algorithms/trust_region.jl")
include("algorithms/extension_algs.jl")

include("utils.jl")
include("default.jl")

# include("function_wrappers.jl")

# @setup_workload begin
#     nlfuncs = ((NonlinearFunction{false}((u, p) -> u .* u .- p), 0.1),
#         (NonlinearFunction{false}((u, p) -> u .* u .- p), [0.1]),
#         (NonlinearFunction{true}((du, u, p) -> du .= u .* u .- p), [0.1]))
#     probs_nls = NonlinearProblem[]
#     for T in (Float32, Float64), (fn, u0) in nlfuncs
#         push!(probs_nls, NonlinearProblem(fn, T.(u0), T(2)))
#     end

#     nls_algs = (NewtonRaphson(), TrustRegion(), LevenbergMarquardt(), PseudoTransient(),
#         Broyden(), Klement(), DFSane(), nothing)

#     probs_nlls = NonlinearLeastSquaresProblem[]
#     nlfuncs = ((NonlinearFunction{false}((u, p) -> (u .^ 2 .- p)[1:1]), [0.1, 0.0]),
#         (NonlinearFunction{false}((u, p) -> vcat(u .* u .- p, u .* u .- p)), [0.1, 0.1]),
#         (NonlinearFunction{true}((du, u, p) -> du[1] = u[1] * u[1] - p,
#                 resid_prototype = zeros(1)), [0.1, 0.0]),
#         (NonlinearFunction{true}((du, u, p) -> du .= vcat(u .* u .- p, u .* u .- p),
#                 resid_prototype = zeros(4)), [0.1, 0.1]))
#     for (fn, u0) in nlfuncs
#         push!(probs_nlls, NonlinearLeastSquaresProblem(fn, u0, 2.0))
#     end
#     nlfuncs = ((NonlinearFunction{false}((u, p) -> (u .^ 2 .- p)[1:1]), Float32[0.1, 0.0]),
#         (NonlinearFunction{false}((u, p) -> vcat(u .* u .- p, u .* u .- p)),
#             Float32[0.1, 0.1]),
#         (NonlinearFunction{true}((du, u, p) -> du[1] = u[1] * u[1] - p,
#                 resid_prototype = zeros(Float32, 1)), Float32[0.1, 0.0]),
#         (NonlinearFunction{true}((du, u, p) -> du .= vcat(u .* u .- p, u .* u .- p),
#                 resid_prototype = zeros(Float32, 4)), Float32[0.1, 0.1]))
#     for (fn, u0) in nlfuncs
#         push!(probs_nlls, NonlinearLeastSquaresProblem(fn, u0, 2.0f0))
#     end

#     nlls_algs = (LevenbergMarquardt(), GaussNewton(),
#         LevenbergMarquardt(; linsolve = LUFactorization()),
#         GaussNewton(; linsolve = LUFactorization()))

#     @compile_workload begin
#         for prob in probs_nls, alg in nls_algs
#             solve(prob, alg, abstol = 1e-2)
#         end
#         for prob in probs_nlls, alg in nlls_algs
#             solve(prob, alg, abstol = 1e-2)
#         end
#     end
# end


# Core Algorithms
export NewtonRaphson, PseudoTransient, Klement, Broyden, LimitedMemoryBroyden, DFSane
export GaussNewton, GradientDescent, LevenbergMarquardt, TrustRegion
# export NonlinearSolvePolyAlgorithm,
#     RobustMultiNewton, FastShortcutNonlinearPolyalg, FastShortcutNLLSPolyalg

# Extension Algorithms
export LeastSquaresOptimJL, FastLevenbergMarquardtJL, CMINPACK, NLsolveJL,
    FixedPointAccelerationJL, SpeedMappingJL, SIAMFANLEquationsJL

# Advanced Algorithms -- Without Bells and Whistles
export GeneralizedFirstOrderAlgorithm, ApproximateJacobianSolveAlgorithm

# Descent Algorithms
export NewtonDescent, SteepestDescent, Dogleg, DampedNewtonDescent,
    GeodesicAcceleration

# Globalization
## Line Search Algorithms
export LineSearchesJL, NoLineSearch, RobustNonMonotoneLineSearch, LiFukushimaLineSearch
## Trust Region Algorithms
export LevenbergMarquardtTrustRegion, RadiusUpdateSchemes, GenericTrustRegionScheme

# Export the termination conditions from DiffEqBase
export SteadyStateDiffEqTerminationMode, SimpleNonlinearSolveTerminationMode,
    NormTerminationMode, RelTerminationMode, RelNormTerminationMode, AbsTerminationMode,
    AbsNormTerminationMode, RelSafeTerminationMode, AbsSafeTerminationMode,
    RelSafeBestTerminationMode, AbsSafeBestTerminationMode

# Tracing Functionality
export TraceAll, TraceMinimal, TraceWithJacobianConditionNumber

end # module
