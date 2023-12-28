module NonlinearSolve

if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end

import Reexport: @reexport
import PrecompileTools: @recompile_invalidations, @compile_workload, @setup_workload

@recompile_invalidations begin
    using ADTypes, DiffEqBase, LazyArrays, LineSearches, LinearAlgebra, LinearSolve, Printf,
        SciMLBase, SimpleNonlinearSolve, SparseArrays, SparseDiffTools, StaticArrays

    import ADTypes: AbstractFiniteDifferencesMode
    import ArrayInterface: undefmatrix, restructure, can_setindex,
        matrix_colors, parameterless_type, ismutable, issingular, fast_scalar_indexing
    import ConcreteStructs: @concrete
    import EnumX: @enumx
    import FastBroadcast: @..
    import FastClosures: @closure
    import FiniteDiff
    import ForwardDiff
    import ForwardDiff: Dual
    import LinearSolve: ComposePreconditioner, InvPreconditioner, needs_concrete_A
    import MaybeInplace: setindex_trait, @bb, CanSetindex, CannotSetindex
    import RecursiveArrayTools: ArrayPartition,
        AbstractVectorOfArray, recursivecopy!, recursivefill!
    import SciMLBase: AbstractNonlinearAlgorithm, NLStats, _unwrap_val, has_jac, isinplace
    import SciMLOperators: FunctionOperator
    import StaticArrays: StaticArray, SVector, SArray, MArray, Size, SMatrix, MMatrix
    import UnPack: @unpack
end

@reexport using ADTypes, LineSearches, SciMLBase, SimpleNonlinearSolve
import DiffEqBase: AbstractNonlinearTerminationMode,
    AbstractSafeNonlinearTerminationMode, AbstractSafeBestNonlinearTerminationMode,
    NonlinearSafeTerminationReturnCode, get_termination_mode

const AbstractSparseADType = Union{ADTypes.AbstractSparseFiniteDifferences,
    ADTypes.AbstractSparseForwardMode, ADTypes.AbstractSparseReverseMode}

import SciMLBase: JacobianWrapper, AbstractNonlinearProblem

import SparseDiffTools: AbstractSparsityDetection

# Type-Inference Friendly Check for Extension Loading
is_extension_loaded(::Val) = false

const True = Val(true)
const False = Val(false)

# abstract type AbstractNonlinearSolveLineSearchAlgorithm end

# abstract type AbstractNewtonAlgorithm{CJ, AD} <: AbstractNonlinearSolveAlgorithm end

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

# function Base.show(io::IO, alg::AbstractNonlinearSolveAlgorithm)
#     str = "$(nameof(typeof(alg)))("
#     modifiers = String[]
#     if __getproperty(alg, Val(:ad)) !== nothing
#         push!(modifiers, "ad = $(nameof(typeof(alg.ad)))()")
#     end
#     if __getproperty(alg, Val(:linsolve)) !== nothing
#         push!(modifiers, "linsolve = $(nameof(typeof(alg.linsolve)))()")
#     end
#     if __getproperty(alg, Val(:linesearch)) !== nothing
#         ls = alg.linesearch
#         if ls isa LineSearch
#             ls.method !== nothing &&
#                 push!(modifiers, "linesearch = $(nameof(typeof(ls.method)))()")
#         else
#             push!(modifiers, "linesearch = $(nameof(typeof(alg.linesearch)))()")
#         end
#     end
#     append!(modifiers, __alg_print_modifiers(alg))
#     if __getproperty(alg, Val(:radius_update_scheme)) !== nothing
#         push!(modifiers, "radius_update_scheme = $(alg.radius_update_scheme)")
#     end
#     str = str * join(modifiers, ", ")
#     print(io, "$(str))")
#     return nothing
# end

# __alg_print_modifiers(_) = String[]

# get_fu(cache::AbstractNonlinearSolveCache) = cache.fu
# set_fu!(cache::AbstractNonlinearSolveCache, fu) = (cache.fu = fu)
# get_u(cache::AbstractNonlinearSolveCache) = cache.u
# SciMLBase.set_u!(cache::AbstractNonlinearSolveCache, u) = (cache.u = u)

include("abstract_types.jl")

include("descent/newton.jl")
include("descent/steepest.jl")
include("descent/dogleg.jl")
include("descent/damped_newton.jl")

include("internal/helpers.jl")
include("internal/jacobian.jl")
# include("internal/forward_diff.jl")
include("internal/linear_solve.jl")
include("internal/operators.jl")
include("internal/termination.jl")
include("internal/tracing.jl")

# include("globalization/damping.jl")
include("globalization/line_search.jl")
# include("globalization/trust_region.jl")

# include("core/approximate_jacobian.jl")
include("core/generalized_first_order.jl")
# include("core/newton.jl")

include("algorithms/raphson.jl")
include("algorithms/gauss_newton.jl")
include("algorithms/pseudo_transient.jl")
# include("algorithms/broyden.jl")
# include("algorithms/klement.jl")

include("utils.jl")
include("default.jl")

# include("function_wrappers.jl")
# include("extension_algs.jl")
# include("trustRegion.jl")
# include("levenberg.jl")
# include("gaussnewton.jl")
# include("dfsane.jl")
# include("pseudotransient.jl")
# include("broyden.jl")
# include("klement.jl")
# include("lbroyden.jl")
# include("jacobian.jl")
# include("ad.jl")
# include("default.jl")

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

# Descent Algorithms
export NewtonDescent, SteepestDescent, Dogleg, DampedNewtonDescent

# Core Algorithms -- Mostly Wrappers
export NewtonRaphson, PseudoTransient
export GaussNewton

# Extension Algorithms

# Advanced Algorithms -- Without Bells and Whistles
export GeneralizedFirstOrderRootFindingAlgorithm

# Line Search Algorithms
export LineSearchesJL, NoLineSearch

# Algorithm Specific Exports
export SwitchedEvolutionRelaxation

# export RadiusUpdateSchemes

# export TrustRegion, LevenbergMarquardt, DFSane,
#     Broyden, Klement, LimitedMemoryBroyden
# export LeastSquaresOptimJL, FastLevenbergMarquardtJL, CMINPACK, NLsolveJL,
#     FixedPointAccelerationJL, SpeedMappingJL, SIAMFANLEquationsJL
# export NonlinearSolvePolyAlgorithm,
#     RobustMultiNewton, FastShortcutNonlinearPolyalg, FastShortcutNLLSPolyalg

# export LineSearch, LiFukushimaLineSearch

# # Export the termination conditions from DiffEqBase
# export SteadyStateDiffEqTerminationMode, SimpleNonlinearSolveTerminationMode,
#     NormTerminationMode, RelTerminationMode, RelNormTerminationMode, AbsTerminationMode,
#     AbsNormTerminationMode, RelSafeTerminationMode, AbsSafeTerminationMode,
#     RelSafeBestTerminationMode, AbsSafeBestTerminationMode

# # Tracing Functionality
# export TraceAll, TraceMinimal, TraceWithJacobianConditionNumber

end # module
