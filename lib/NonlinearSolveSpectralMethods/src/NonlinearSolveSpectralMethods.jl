"""
    NonlinearSolveSpectralMethods

Spectral residual methods for nonlinear systems.

This subpackage provides `DFSane` and the generalized spectral residual
implementation used by NonlinearSolve.jl. Use it for derivative-free nonlinear
solves where a spectral residual method is appropriate.

### Example

```julia
using NonlinearSolveSpectralMethods, SciMLBase

prob = NonlinearProblem((u, p) -> u^2 - p, 1.0, 2.0)
sol = solve(prob, DFSane())
```
"""
module NonlinearSolveSpectralMethods

using ConcreteStructs: @concrete
using Reexport: @reexport
using PrecompileTools: @compile_workload, @setup_workload

using CommonSolve: CommonSolve
using LineSearch: RobustNonMonotoneLineSearch
using MaybeInplace: @bb
using NonlinearSolveBase: NonlinearSolveBase, AbstractNonlinearSolveAlgorithm,
    AbstractNonlinearSolveCache, Utils, InternalAPI, get_timer_output,
    @static_timeit, update_trace!, NonlinearVerbosity
using SciMLBase: SciMLBase, AbstractNonlinearProblem, NLStats, ReturnCode,
    NonlinearProblem, NonlinearFunction, NoSpecialize
using SciMLLogging: None, AbstractVerbosityPreset

include("dfsane.jl")

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

    algs = [DFSane()]

    @compile_workload begin
        @sync for prob in nonlinear_problems, alg in algs

            Threads.@spawn CommonSolve.solve(prob, alg; abstol = 1.0e-2, verbose = false)
        end
    end
end

@reexport using SciMLBase, NonlinearSolveBase

export GeneralizedDFSane, DFSane

end
