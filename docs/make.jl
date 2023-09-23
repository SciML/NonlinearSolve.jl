using Documenter, NonlinearSolve, SimpleNonlinearSolve, Sundials, SciMLNLSolve,
    NonlinearSolveMINPACK, SteadyStateDiffEq

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

include("pages.jl")

makedocs(sitename = "NonlinearSolve.jl",
    authors = "Chris Rackauckas",
    modules = [NonlinearSolve, NonlinearSolve.SciMLBase, NonlinearSolve.DiffEqBase,
        SimpleNonlinearSolve, Sundials, SciMLNLSolve, NonlinearSolveMINPACK,
        SteadyStateDiffEq],
    clean = true, doctest = false, linkcheck = true,
    linkcheck_ignore = ["https://twitter.com/ChrisRackauckas/status/1544743542094020615"],
    warnonly = [:missing_docs, :cross_references],
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/NonlinearSolve/stable/"),
    pages = pages)

deploydocs(repo = "github.com/SciML/NonlinearSolve.jl.git";
    push_preview = true)
