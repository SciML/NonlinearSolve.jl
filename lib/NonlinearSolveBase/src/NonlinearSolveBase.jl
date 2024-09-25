module NonlinearSolveBase

using ADTypes: ADTypes, AbstractADType, ForwardMode, ReverseMode
using ArrayInterface: ArrayInterface
using Compat: @compat
using ConcreteStructs: @concrete
using DifferentiationInterface: DifferentiationInterface
using EnzymeCore: EnzymeCore
using FastClosures: @closure
using LinearAlgebra: norm
using Markdown: @doc_str
using RecursiveArrayTools: AbstractVectorOfArray, ArrayPartition
using SciMLBase: SciMLBase, ReturnCode, AbstractODEIntegrator, AbstractNonlinearProblem,
                 NonlinearProblem, NonlinearLeastSquaresProblem, AbstractNonlinearFunction,
                 @add_kwonly, StandardNonlinearProblem, NullParameters, NonlinearProblem,
                 isinplace, warn_paramtype
using StaticArraysCore: StaticArray

const DI = DifferentiationInterface

include("public.jl")
include("utils.jl")

include("immutable_problem.jl")
include("common_defaults.jl")
include("termination_conditions.jl")

include("autodiff.jl")

# Unexported Public API
@compat(public, (L2_NORM, Linf_NORM, NAN_CHECK, UNITLESS_ABS2, get_tolerance))
@compat(public, (nonlinearsolve_forwarddiff_solve, nonlinearsolve_dual_solution))
@compat(public, (select_forward_mode_autodiff, select_reverse_mode_autodiff,
    select_jacobian_autodiff))

export RelTerminationMode, AbsTerminationMode, NormTerminationMode, RelNormTerminationMode,
       AbsNormTerminationMode, RelNormSafeTerminationMode, AbsNormSafeTerminationMode,
       RelNormSafeNormTerminationMode, AbsNormSafeNormTerminationMode

end
