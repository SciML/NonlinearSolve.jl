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
using LinearAlgebra: LinearAlgebra, Diagonal, dot
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
using Setfield: @set!
using StaticArraysCore: SArray

include("raphson.jl")
include("gauss_newton.jl")
include("levenberg_marquardt.jl")
include("trust_region.jl")
include("pseudo_transient.jl")

include("solve.jl")

@setup_workload begin
    include(joinpath(
        @__DIR__, "..", "..", "..", "common", "nonlinear_problem_workloads.jl"
    ))
    include(joinpath(
        @__DIR__, "..", "..", "..", "common", "nlls_problem_workloads.jl"
    ))

    # XXX: TrustRegion
    nlp_algs = [NewtonRaphson(), LevenbergMarquardt()]
    nlls_algs = [GaussNewton(), LevenbergMarquardt()]

    @compile_workload begin
        for prob in nonlinear_problems, alg in nlp_algs
            CommonSolve.solve(prob, alg; abstol = 1e-2, verbose = false)
        end

        for prob in nlls_problems, alg in nlls_algs
            CommonSolve.solve(prob, alg; abstol = 1e-2, verbose = false)
        end
    end
end

@reexport using SciMLBase, NonlinearSolveBase

export NewtonRaphson, PseudoTransient
export GaussNewton, LevenbergMarquardt, TrustRegion

export GeneralizedFirstOrderAlgorithm

end
