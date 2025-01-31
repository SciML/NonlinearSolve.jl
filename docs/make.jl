using Documenter, DocumenterCitations, DocumenterInterLinks
import DiffEqBase

using Sundials
using NonlinearSolveBase, SciMLBase, DiffEqBase
using SimpleNonlinearSolve, BracketingNonlinearSolve
using NonlinearSolveFirstOrder, NonlinearSolveQuasiNewton, NonlinearSolveSpectralMethods
using NonlinearSolveHomotopyContinuation
using SciMLJacobianOperators
using NonlinearSolve, SteadyStateDiffEq

cp(
    joinpath(@__DIR__, "Manifest.toml"),
    joinpath(@__DIR__, "src/assets/Manifest.toml");
    force = true
)
cp(
    joinpath(@__DIR__, "Project.toml"),
    joinpath(@__DIR__, "src/assets/Project.toml");
    force = true
)

include("pages.jl")

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

interlinks = InterLinks(
    "ADTypes" => "https://sciml.github.io/ADTypes.jl/stable/",
    "LineSearch" => "https://sciml.github.io/LineSearch.jl/dev/"
)

makedocs(;
    sitename = "NonlinearSolve.jl",
    authors = "SciML",
    modules = [
        NonlinearSolveBase, SciMLBase, DiffEqBase,
        SimpleNonlinearSolve, BracketingNonlinearSolve,
        NonlinearSolveFirstOrder, NonlinearSolveQuasiNewton, NonlinearSolveSpectralMethods,
        NonlinearSolveHomotopyContinuation,
        Sundials,
        SciMLJacobianOperators,
        NonlinearSolve, SteadyStateDiffEq
    ],
    clean = true,
    doctest = false,
    linkcheck = true,
    linkcheck_ignore = [
        "https://twitter.com/ChrisRackauckas/status/1544743542094020615",
        "https://link.springer.com/article/10.1007/s40096-020-00339-4"
    ],
    checkdocs = :exports,
    warnonly = [:missing_docs],
    plugins = [bib, interlinks],
    format = Documenter.HTML(
        assets = ["assets/favicon.ico", "assets/citations.css"],
        canonical = "https://docs.sciml.ai/NonlinearSolve/stable/"
    ),
    pages
)

deploydocs(repo = "github.com/SciML/NonlinearSolve.jl.git"; push_preview = true)
