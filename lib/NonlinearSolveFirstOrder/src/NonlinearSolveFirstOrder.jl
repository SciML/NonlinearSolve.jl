module NonlinearSolveFirstOrder

using ConcreteStructs: @concrete
using PrecompileTools: @compile_workload, @setup_workload
using Reexport: @reexport
using Setfield: @set!

using ADTypes: ADTypes
using ArrayInterface: ArrayInterface
using LinearAlgebra: LinearAlgebra, Diagonal, dot, diagind
using LineSearch: BackTracking
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
                          update_trace!, L2_NORM, NonlinearSolvePolyAlgorithm,
                          NewtonDescent, DampedNewtonDescent, GeodesicAcceleration,
                          Dogleg
using SciMLBase: SciMLBase, AbstractNonlinearProblem, NLStats, ReturnCode,
                 NonlinearFunction,
                 NonlinearLeastSquaresProblem, NonlinearProblem, NoSpecialize
using SciMLJacobianOperators: VecJacOperator, JacVecOperator, StatefulJacobianOperator

using FiniteDiff: FiniteDiff    # Default Finite Difference Method
using ForwardDiff: ForwardDiff  # Default Forward Mode AD

include("raphson.jl")
include("gauss_newton.jl")
include("levenberg_marquardt.jl")
include("trust_region.jl")
include("pseudo_transient.jl")

include("poly_algs.jl")

include("solve.jl")

@setup_workload begin
    nonlinear_functions = (
        (NonlinearFunction{false, NoSpecialize}((u, p) -> u .* u .- p), 0.1),
        (NonlinearFunction{false, NoSpecialize}((u, p) -> u .* u .- p), [0.1]),
        (NonlinearFunction{true, NoSpecialize}((du, u, p) -> du .= u .* u .- p), [0.1])
    )

    nonlinear_problems = NonlinearProblem[]
    for (fn, u0) in nonlinear_functions
        push!(nonlinear_problems, NonlinearProblem(fn, u0, 2.0))
    end

    nonlinear_functions = (
        (NonlinearFunction{false, NoSpecialize}((u, p) -> (u .^ 2 .- p)[1:1]), [0.1, 0.0]),
        (
            NonlinearFunction{false, NoSpecialize}((u, p) -> vcat(u .* u .- p, u .* u .- p)),
            [0.1, 0.1]
        ),
        (
            NonlinearFunction{true, NoSpecialize}(
                (du, u, p) -> du[1] = u[1] * u[1] - p, resid_prototype = zeros(1)
            ),
            [0.1, 0.0]
        ),
        (
            NonlinearFunction{true, NoSpecialize}(
                (du, u, p) -> du .= vcat(u .* u .- p, u .* u .- p), resid_prototype = zeros(4)
            ),
            [0.1, 0.1]
        )
    )

    nlls_problems = NonlinearLeastSquaresProblem[]
    for (fn, u0) in nonlinear_functions
        push!(nlls_problems, NonlinearLeastSquaresProblem(fn, u0, 2.0))
    end

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

# Polyalgorithms
export RobustMultiNewton

end
