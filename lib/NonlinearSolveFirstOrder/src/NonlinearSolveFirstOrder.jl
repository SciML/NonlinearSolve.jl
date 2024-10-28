module NonlinearSolveFirstOrder

using Reexport: @reexport
using PrecompileTools: @compile_workload, @setup_workload

using ADTypes: ADTypes
using ArrayInterface: ArrayInterface
using CommonSolve: CommonSolve
using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase    # Needed for `init` / `solve` dispatches
using FiniteDiff: FiniteDiff    # Default Finite Difference Method
using ForwardDiff: ForwardDiff  # Default Forward Mode AD
using LinearAlgebra: LinearAlgebra, Diagonal, dot, inv, diag
using LinearSolve: LinearSolve  # Trigger Linear Solve extension in NonlinearSolveBase
using MaybeInplace: @bb
using NonlinearSolveBase: NonlinearSolveBase, AbstractNonlinearSolveAlgorithm,
                          AbstractNonlinearSolveCache, AbstractResetCondition,
                          AbstractResetConditionCache, AbstractApproximateJacobianStructure,
                          AbstractJacobianCache, AbstractJacobianInitialization,
                          AbstractApproximateJacobianUpdateRule, AbstractDescentDirection,
                          AbstractApproximateJacobianUpdateRuleCache,
                          AbstractDampingFunction, AbstractDampingFunctionCache,
                          AbstractTrustRegionMethod, AbstractTrustRegionMethodCache,
                          Utils, InternalAPI, get_timer_output, @static_timeit,
                          update_trace!, L2_NORM,
                          NewtonDescent, DampedNewtonDescent
using SciMLBase: SciMLBase, AbstractNonlinearProblem, NLStats, ReturnCode
using SciMLOperators: AbstractSciMLOperator
using Setfield: @set!
using StaticArraysCore: StaticArray, SArray, Size, MArray

include("raphson.jl")
include("gauss_newton.jl")
include("levenberg_marquardt.jl")
include("trust_region.jl")
include("pseudo_transient.jl")

include("solve.jl")

@reexport using SciMLBase, NonlinearSolveBase

export NewtonRaphson, PseudoTransient
export GaussNewton, LevenbergMarquardt, TrustRegion

export GeneralizedFirstOrderAlgorithm

end
