# NonlinearSolveSciPy.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/NonlinearSolve/stable/)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

NonlinearSolveSciPy.jl is a component of the [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) monorepo. It wraps SciPy's `scipy.optimize` root-finding and least-squares routines (via PythonCall.jl) behind the SciML common interface.
While completely independent and usable on its own, users wanting the full nonlinear solver suite should use [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl).

For information on using the package, see the [stable documentation](https://docs.sciml.ai/NonlinearSolve/stable/). Use the [in-development documentation](https://docs.sciml.ai/NonlinearSolve/dev/) for the version of the documentation which contains the unreleased features.
