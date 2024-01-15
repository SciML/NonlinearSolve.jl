using Documenter, DocumenterCitations
using NonlinearSolve,
    SimpleNonlinearSolve, Sundials, SteadyStateDiffEq, SciMLBase, DiffEqBase

cp(joinpath(@__DIR__, "Manifest.toml"), joinpath(@__DIR__, "src/assets/Manifest.toml"),
    force = true)
cp(joinpath(@__DIR__, "Project.toml"), joinpath(@__DIR__, "src/assets/Project.toml"),
    force = true)

include("pages.jl")

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(; sitename = "NonlinearSolve.jl",
    authors = "Chris Rackauckas",
    modules = [NonlinearSolve, SimpleNonlinearSolve, SteadyStateDiffEq, Sundials,
        DiffEqBase, SciMLBase],
    clean = true, doctest = false, linkcheck = true,
    linkcheck_ignore = ["https://twitter.com/ChrisRackauckas/status/1544743542094020615"],
    checkdocs = :exports, warnonly = false, plugins = [bib],
    format = Documenter.HTML(assets = ["assets/favicon.ico", "assets/citations.css"],
        canonical = "https://docs.sciml.ai/NonlinearSolve/stable/"),
    pages)

deploydocs(repo = "github.com/SciML/NonlinearSolve.jl.git"; push_preview = true)
