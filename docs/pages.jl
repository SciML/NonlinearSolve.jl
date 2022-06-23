# Put in a separate page so it can be used by SciMLDocs.jl

pages = ["Home" => "index.md",
         "Tutorials" => Any["tutorials/nonlinear.md",
                            "tutorials/iterator_interface.md"],
         "Basics" => Any["basics/NonlinearProblem.md",
                         "basics/NonlinearFunctions.md",
                         "basics/FAQ.md"],
         "Solvers" => Any["solvers/NonlinearSystemSolvers.md",
                          "solvers/BracketingSolvers.md"]]
