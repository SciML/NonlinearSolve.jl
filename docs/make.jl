using Documenter, NonlinearSolve

include("pages.jl")

makedocs(sitename = "NonlinearSolve.jl",
         authors = "Chris Rackauckas",
         modules = [NonlinearSolve, NonlinearSolve.SciMLBase],
         clean = true, doctest = false,
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://nonlinearsolve.sciml.ai/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/NonlinearSolve.jl.git";
           push_preview = true)
