using Documenter, NonlinearSolve

makedocs(
    sitename="NonlinearSolve.jl",
    authors="Chris Rackauckas",
    modules=[NonlinearSolve],
    clean=true,doctest=false,
    format = Documenter.HTML(#analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://nlsolve.sciml.ai/stable/"),
    pages=[
        "Home" => "index.md",
        "Tutorials" => Any[
            "tutorials/nonlinear.md"
        ],
        "Basics" => Any[
            "basics/NonlinearProblem.md",
            "basics/NonlinearFunctions.md",
            "basics/FAQ.md"
        ],
        "Solvers" => Any[
            "solvers/NonlinearSystemSolvers.md",
            "solvers/BracketingSolvers.md"
        ]
    ]
)

deploydocs(
   repo = "github.com/SciML/NonlinearSolve.jl.git";
   push_preview = true
)
