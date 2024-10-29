module NonlinearSolveSpectralMethods

using ConcreteStructs: @concrete
using Reexport: @reexport
using PrecompileTools: @compile_workload, @setup_workload

using CommonSolve: CommonSolve
using DiffEqBase: DiffEqBase # Needed for `init` / `solve` dispatches
using LineSearch: RobustNonMonotoneLineSearch
using MaybeInplace: @bb
using NonlinearSolveBase: NonlinearSolveBase, AbstractNonlinearSolveAlgorithm,
                          AbstractNonlinearSolveCache, Utils, InternalAPI, get_timer_output,
                          @static_timeit, update_trace!
using SciMLBase: SciMLBase, AbstractNonlinearProblem, NLStats, ReturnCode

include("dfsane.jl")

include("solve.jl")

@setup_workload begin
    include("../../../common/nonlinear_problem_workloads.jl")

    algs = [DFSane()]

    @compile_workload begin
        @sync for prob in nonlinear_problems, alg in algs
            Threads.@spawn CommonSolve.solve(prob, alg; abstol = 1e-2, verbose = false)
        end
    end
end

@reexport using SciMLBase, NonlinearSolveBase

export GeneralizedDFSane, DFSane

end
