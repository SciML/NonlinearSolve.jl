  - This repository follows the [SciMLStyle](https://github.com/SciML/SciMLStyle) and the SciML [ColPrac](https://github.com/SciML/ColPrac).
  - Please run `using JuliaFormatter, NonlinearSolve; format(joinpath(dirname(pathof(NonlinearSolve)), ".."))` before committing.
  - Add tests for any new features.
  - Additional help on contributing to the numerical solvers can be found at https://docs.sciml.ai/NonlinearSolve/stable/

## Developing Locally

NonlinearSolve is a large package and thus it uses a sublibrary approach to keep down
the total number of dependencies per solver. As a consequence, it requires a bit of special
handling compared to some other Julia packages.

When running the subpackage test suite, it's recommended that one has dev'd its relevant packages in NonlinearSolve.jl. This can be done via(use `NonlinearSolveFirstOrder` as example):

```julia
using Pkg
dev_pks = Pkg.PackageSpec[]
for path in ("lib/SciMLJacobianOperators", "lib/NonlinearSolveBase")
    push!(dev_pks, Pkg.PackageSpec(; path))
end
Pkg.develop(dev_pks)
```

When running the full test suite, it's
recommended that one has dev'd all of the relevant packages. This can be done via:

```julia
using Pkg
Pkg.develop(map(
    path -> Pkg.PackageSpec.(; path = "$(@__DIR__)/lib/$(path)"), readdir("./lib")));
```

and then running tests accordingly.

## Dependency Structure

There is a tree dependency structure to the sublibraries of NonlinearSolve.jl. The core subpackage NonlinearSolveBase.jl is the hard dependency of all the other subpackages. SimpleNonlinearSolve.jl is a special one with BracketingNonlinearSolve.jl as dependency. NonlinearSolve.jl as the parent package contains all of the subpackages.
