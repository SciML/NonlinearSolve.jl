using Documenter,
    NonlinearSolve, SimpleNonlinearSolve, Sundials, SteadyStateDiffEq, SciMLBase, DiffEqBase

cp(joinpath(@__DIR__, "Manifest.toml"), joinpath(@__DIR__, "src/assets/Manifest.toml"),
    force = true)
cp(joinpath(@__DIR__, "Project.toml"), joinpath(@__DIR__, "src/assets/Project.toml"),
    force = true)

include("pages.jl")

makedocs(; sitename = "NonlinearSolve.jl",
    authors = "Chris Rackauckas",
    modules = [NonlinearSolve, SimpleNonlinearSolve, SteadyStateDiffEq, Sundials,
        DiffEqBase, SciMLBase],
    clean = true, doctest = false, linkcheck = true,
    linkcheck_ignore = ["https://twitter.com/ChrisRackauckas/status/1544743542094020615"],
    checkdocs = :exports,
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/NonlinearSolve/stable/"),
    pages)

deploydocs(repo = "github.com/SciML/NonlinearSolve.jl.git"; push_preview = true)
