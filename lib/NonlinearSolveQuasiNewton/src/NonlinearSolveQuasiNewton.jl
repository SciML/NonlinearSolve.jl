module NonlinearSolveQuasiNewton

using ConcreteStructs: @concrete
using PrecompileTools: @compile_workload, @setup_workload
using Reexport: @reexport

using ArrayInterface: ArrayInterface
using StaticArraysCore: StaticArray, Size, MArray

using CommonSolve: CommonSolve
using LinearAlgebra: LinearAlgebra, Diagonal, dot, diag
using LinearSolve: LinearSolve # Trigger Linear Solve extension in NonlinearSolveBase
using MaybeInplace: @bb
using NonlinearSolveBase: NonlinearSolveBase, AbstractNonlinearSolveAlgorithm,
                          AbstractNonlinearSolveCache, AbstractResetCondition,
                          AbstractResetConditionCache, AbstractApproximateJacobianStructure,
                          AbstractJacobianCache, AbstractJacobianInitialization,
                          AbstractApproximateJacobianUpdateRule, AbstractDescentDirection,
                          AbstractApproximateJacobianUpdateRuleCache,
                          Utils, InternalAPI, get_timer_output, @static_timeit,
                          update_trace!, L2_NORM, NewtonDescent
using SciMLBase: SciMLBase, AbstractNonlinearProblem, NLStats, ReturnCode,
                 NonlinearProblem, NonlinearFunction, NoSpecialize
using SciMLOperators: AbstractSciMLOperator

include("reset_conditions.jl")
include("structure.jl")
include("initialization.jl")

include("broyden.jl")
include("lbroyden.jl")
include("klement.jl")

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

    algs = [Broyden(), Klement()]

    @compile_workload begin
        @sync for prob in nonlinear_problems, alg in algs

            Threads.@spawn CommonSolve.solve(prob, alg; abstol = 1e-2, verbose = false)
        end
    end
end

@reexport using SciMLBase, NonlinearSolveBase

export Broyden, LimitedMemoryBroyden, Klement, QuasiNewtonAlgorithm

end
