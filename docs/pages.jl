# Put in a separate page so it can be used by SciMLDocs.jl

pages = ["index.md",
    "Tutorials" => Any["tutorials/nonlinear.md",
                       "tutorials/iterator_interface.md"],
    "Basics" => Any["basics/NonlinearProblem.md",
                    "basics/NonlinearFunctions.md",
                    "basics/NonlinearSolution.md",
                    "basics/FAQ.md"],
    "Solver Summaries and Recommendations" => Any[
                     "solvers/NonlinearSystemSolvers.md",
                     "solvers/BracketingSolvers.md",
                     "solvers/SteadyStateSolvers.md"],
    "Detailed Solver APIs" => Any[
        "nonlinearsolve.md",
        "simplenonlinearsolve.md",
        "minpack.md",
        "nlsolve.md",
        "sundials.md",
        "steadystatediffeq.md"
    ]
]
