module NonlinearSolveBase

using Compat: @compat
using ConcreteStructs: @concrete
using FastClosures: @closure
using Preferences: @load_preference, @set_preferences!

using ADTypes: ADTypes, AbstractADType, AutoSparse, AutoForwardDiff, NoSparsityDetector,
               KnownJacobianSparsityDetector
using Adapt: WrappedArray
using ArrayInterface: ArrayInterface
using DifferentiationInterface: DifferentiationInterface, Constant
using StaticArraysCore: StaticArray, SMatrix, SArray, MArray

using CommonSolve: CommonSolve, init
using EnzymeCore: EnzymeCore
using MaybeInplace: @bb
using RecursiveArrayTools: RecursiveArrayTools, AbstractVectorOfArray, ArrayPartition
using SciMLBase: SciMLBase, ReturnCode, AbstractODEIntegrator, AbstractNonlinearProblem,
                 AbstractNonlinearAlgorithm, _concrete_solve_adjoint, _concrete_solve_forward,
                 NonlinearProblem, NonlinearLeastSquaresProblem,
                 NonlinearFunction, NLStats, LinearProblem,
                 LinearAliasSpecifier, ImmutableNonlinearProblem, NonlinearAliasSpecifier,
                 promote_u0, get_concrete_u0, get_concrete_p,
                 has_kwargs, extract_alg, promote_u0, checkkwargs, SteadyStateProblem,
                 NoDefaultAlgorithmError, NonSolverError, KeywordArgError, AbstractDEAlgorithm
import SciMLBase: solve, init, __init, __solve, wrap_sol, get_root_indp, isinplace, remake

using SciMLJacobianOperators: JacobianOperator, StatefulJacobianOperator
using SciMLOperators: AbstractSciMLOperator, IdentityOperator
using SciMLLogging: @SciMLMessage, AbstractVerbositySpecifier, AbstractVerbosityPreset, AbstractMessageLevel, 
                None, Minimal, Standard, Detailed, All, Silent, InfoLevel, WarnLevel

using SymbolicIndexingInterface: SymbolicIndexingInterface
import SciMLStructures
using Setfield: @set!

using LinearAlgebra: LinearAlgebra, Diagonal, norm, ldiv!, diagind, mul!
using Markdown: @doc_str
using Printf: @printf

const DI = DifferentiationInterface
const SII = SymbolicIndexingInterface

# Custom keyword argument handler that extends the standard SciMLBase keywords
# to include bounds (lb, ub) for NonlinearLeastSquaresProblem
struct NonlinearKeywordArgError end

function SciMLBase.checkkwargs(::Type{NonlinearKeywordArgError}; kwargs...)
    keywords = keys(kwargs)
    allowed_keywords = (:dense, :saveat, :save_idxs, :save_discretes, :tstops, :tspan,
                       :d_discontinuities, :save_everystep, :save_on, :save_start, :save_end,
                       :initialize_save, :adaptive, :abstol, :reltol, :dt, :dtmax, :dtmin,
                       :force_dtmin, :internalnorm, :controller, :gamma, :beta1, :beta2,
                       :qmax, :qmin, :qsteady_min, :qsteady_max, :qoldinit, :failfactor,
                       :calck, :alias_u0, :maxiters, :maxtime, :callback, :isoutofdomain,
                       :unstable_check, :verbose, :merge_callbacks, :progress, :progress_steps,
                       :progress_name, :progress_message, :progress_id, :timeseries_errors,
                       :dense_errors, :weak_timeseries_errors, :weak_dense_errors, :wrap,
                       :calculate_error, :initializealg, :alg, :save_noise, :delta, :seed,
                       :alg_hints, :kwargshandle, :trajectories, :batch_size, :sensealg,
                       :advance_to_tstop, :stop_at_next_tstop, :u0, :p, :default_set,
                       :second_time, :prob_choice, :alias_jump, :alias_noise, :batch,
                       :nlsolve_kwargs, :odesolve_kwargs, :linsolve_kwargs, :ensemblealg,
                       :show_trace, :trace_level, :store_trace, :termination_condition,
                       :alias, :fit_parameters, :lb, :ub)  # Added lb and ub
    for kw in keywords
        if kw ∉ allowed_keywords
            throw(SciMLBase.KeywordArgumentError(kw))
        end
    end
end

include("public.jl")
include("utils.jl")
include("verbosity.jl")

include("abstract_types.jl")
include("common_defaults.jl")
include("termination_conditions.jl")

include("autodiff.jl")
include("jacobian.jl")
include("linear_solve.jl")
include("timer_outputs.jl")
include("tracing.jl")
include("wrappers.jl")
include("polyalg.jl")


include("descent/common.jl")
include("descent/newton.jl")
include("descent/steepest.jl")
include("descent/damped_newton.jl")
include("descent/dogleg.jl")
include("descent/geodesic_acceleration.jl")

include("initialization.jl")
include("solve.jl")

include("forward_diff.jl")

# Unexported Public API
@compat(public, (L2_NORM, Linf_NORM, NAN_CHECK, UNITLESS_ABS2, get_tolerance))
@compat(public, (nonlinearsolve_forwarddiff_solve, nonlinearsolve_dual_solution))
@compat(public,
    (select_forward_mode_autodiff, select_reverse_mode_autodiff, select_jacobian_autodiff))

# public for NonlinearSolve.jl and subpackages to use
@compat(public, (InternalAPI, supports_line_search, supports_trust_region, set_du!))
@compat(public, (construct_linear_solver, needs_square_A, needs_concrete_A))
@compat(public, (construct_jacobian_cache, reused_jacobian))
@compat(public,
    (assert_extension_supported_termination_condition,
    construct_extension_function_wrapper, construct_extension_jac))

export TraceMinimal, TraceWithJacobianConditionNumber, TraceAll

export RelTerminationMode, AbsTerminationMode,
       NormTerminationMode, RelNormTerminationMode, AbsNormTerminationMode,
       RelNormSafeTerminationMode, AbsNormSafeTerminationMode,
       RelNormSafeBestTerminationMode, AbsNormSafeBestTerminationMode

export DescentResult, SteepestDescent, NewtonDescent, DampedNewtonDescent, Dogleg,
       GeodesicAcceleration

export NonlinearSolvePolyAlgorithm

export NonlinearVerbosity

export pickchunksize

end
