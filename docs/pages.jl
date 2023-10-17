# Put in a separate page so it can be used by SciMLDocs.jl

pages = ["index.md",
    "tutorials/getting_started.md"
    "Tutorials" => Any[
        "tutorials/code_optimization.md",
        "tutorials/large_systems.md",
        "tutorials/small_compile.md",
        "tutorials/termination_conditions.md",
        "tutorials/iterator_interface.md"],
    "Basics" => Any["basics/NonlinearProblem.md",
        "basics/NonlinearFunctions.md",
        "basics/solve.md",
        "basics/NonlinearSolution.md",
        "basics/TerminationCondition.md",
        "basics/FAQ.md"],
    "Solver Summaries and Recommendations" => Any["solvers/NonlinearSystemSolvers.md",
        "solvers/BracketingSolvers.md",
        "solvers/SteadyStateSolvers.md",
        "solvers/NonlinearLeastSquaresSolvers.md"],
    "Detailed Solver APIs" => Any["api/nonlinearsolve.md",
        "api/simplenonlinearsolve.md",
        "api/minpack.md",
        "api/nlsolve.md",
        "api/sundials.md",
        "api/steadystatediffeq.md"],
]
