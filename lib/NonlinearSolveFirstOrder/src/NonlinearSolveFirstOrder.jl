module NonlinearSolveFirstOrder

using ConcreteStructs: @concrete
using PrecompileTools: @compile_workload, @setup_workload
using Reexport: @reexport
using Setfield: @set!

using ADTypes: ADTypes
using ArrayInterface: ArrayInterface
using LinearAlgebra: LinearAlgebra, Diagonal, dot
using StaticArraysCore: SArray

using CommonSolve: CommonSolve
using DiffEqBase: DiffEqBase    # Needed for `init` / `solve` dispatches
using LinearSolve: LinearSolve  # Trigger Linear Solve extension in NonlinearSolveBase
using MaybeInplace: @bb
using NonlinearSolveBase: NonlinearSolveBase, AbstractNonlinearSolveAlgorithm,
                          AbstractNonlinearSolveCache, AbstractDampingFunction,
                          AbstractDampingFunctionCache, AbstractTrustRegionMethod,
                          AbstractTrustRegionMethodCache,
                          Utils, InternalAPI, get_timer_output, @static_timeit,
                          update_trace!, L2_NORM,
                          NewtonDescent, DampedNewtonDescent, GeodesicAcceleration,
                          Dogleg
using SciMLBase: SciMLBase, AbstractNonlinearProblem, NLStats, ReturnCode, NonlinearFunction
using SciMLJacobianOperators: VecJacOperator, JacVecOperator, StatefulJacobianOperator

using FiniteDiff: FiniteDiff    # Default Finite Difference Method
using ForwardDiff: ForwardDiff  # Default Forward Mode AD

include("raphson.jl")
include("gauss_newton.jl")
include("levenberg_marquardt.jl")
include("trust_region.jl")
include("pseudo_transient.jl")

include("solve.jl")

@setup_workload begin
    include("../../../common/nonlinear_problem_workloads.jl")
    include("../../../common/nlls_problem_workloads.jl")

    nlp_algs = [NewtonRaphson(), TrustRegion(), LevenbergMarquardt()]
    nlls_algs = [GaussNewton(), TrustRegion(), LevenbergMarquardt()]

    @compile_workload begin
        @sync begin
            for prob in nonlinear_problems, alg in nlp_algs
                Threads.@spawn CommonSolve.solve(prob, alg; abstol = 1e-2, verbose = false)
            end

            for prob in nlls_problems, alg in nlls_algs
                Threads.@spawn CommonSolve.solve(prob, alg; abstol = 1e-2, verbose = false)
            end
        end
    end
end

@reexport using SciMLBase, NonlinearSolveBase

export NewtonRaphson, PseudoTransient
export GaussNewton, LevenbergMarquardt, TrustRegion

export RadiusUpdateSchemes

export GeneralizedFirstOrderAlgorithm

end
