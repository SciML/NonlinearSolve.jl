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
using RecursiveArrayTools: AbstractVectorOfArray, ArrayPartition
using SciMLBase: SciMLBase, ReturnCode, AbstractODEIntegrator, AbstractNonlinearProblem,
                 AbstractNonlinearAlgorithm, AbstractNonlinearFunction,
                 NonlinearProblem, NonlinearLeastSquaresProblem, StandardNonlinearProblem,
                 NonlinearFunction, NullParameters, NLStats, LinearProblem, LinearAliasSpecifier
using SciMLJacobianOperators: JacobianOperator, StatefulJacobianOperator
using SciMLOperators: AbstractSciMLOperator, IdentityOperator
using SymbolicIndexingInterface: SymbolicIndexingInterface

using LinearAlgebra: LinearAlgebra, Diagonal, norm, ldiv!, diagind, mul!
using Markdown: @doc_str
using Printf: @printf

const DI = DifferentiationInterface
const SII = SymbolicIndexingInterface

include("public.jl")
include("utils.jl")

include("abstract_types.jl")

include("immutable_problem.jl")
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
@compat(public, (construct_jacobian_cache,))
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

export pickchunksize

end
