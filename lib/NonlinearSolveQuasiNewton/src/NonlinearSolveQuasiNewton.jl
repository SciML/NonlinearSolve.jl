"""
    NonlinearSolveQuasiNewton

Quasi-Newton nonlinear solver algorithms.

This subpackage implements quasi-Newton methods such as `Broyden`, `Klement`, and
`LimitedMemoryBroyden`. These algorithms are re-exported by NonlinearSolve.jl, but
the subpackage can also be loaded directly by packages that only need quasi-Newton
solver implementations.

### Example

```julia
using NonlinearSolveQuasiNewton, SciMLBase

prob = NonlinearProblem((u, p) -> u^2 - p, 1.0, 2.0)
sol = solve(prob, Broyden())
```
"""
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
    update_trace!, L2_NORM, NewtonDescent, NonlinearVerbosity, reused_jacobian
using SciMLBase: SciMLBase, AbstractNonlinearProblem, NLStats, ReturnCode,
    NonlinearProblem, NonlinearFunction, NoSpecialize
using SciMLLogging: @SciMLMessage, None, AbstractVerbosityPreset
using SciMLOperators: SciMLOperators, AbstractSciMLOperator

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
        (NonlinearFunction{true, NoSpecialize}((du, u, p) -> du .= u .* u .- p), [0.1]),
    )

    nonlinear_problems = NonlinearProblem[]
    for (fn, u0) in nonlinear_functions
        push!(nonlinear_problems, NonlinearProblem(fn, u0, 2.0))
    end

    # AutoDePSpecialize opaque-p path: an isbits `p` packs into an `OpaqueParams`
    # and a non-isbits `p` into an `OpaqueRef`, each a single wrapped-residual
    # signature shared across all parameter types of that kind, so one
    # precompiled solve per container serves first solves with struct/array `p`.
    push!(
        nonlinear_problems, NonlinearProblem(
            NonlinearFunction{true, SciMLBase.AutoDePSpecialize}(
                (du, u, p) -> (du .= u .* u .- p.a)
            ), [0.1], (a = 2.0,)
        )
    )
    push!(
        nonlinear_problems, NonlinearProblem(
            NonlinearFunction{true, SciMLBase.AutoDePSpecialize}(
                (du, u, p) -> (du .= u .* u .- p[1])
            ), [0.1], [2.0]
        )
    )

    algs = [Broyden(), Klement()]

    @compile_workload begin
        @sync for prob in nonlinear_problems, alg in algs

            Threads.@spawn CommonSolve.solve(prob, alg; abstol = 1.0e-2, verbose = false)
        end
    end
end

@reexport using SciMLBase, NonlinearSolveBase

export Broyden, LimitedMemoryBroyden, Klement, QuasiNewtonAlgorithm

end
