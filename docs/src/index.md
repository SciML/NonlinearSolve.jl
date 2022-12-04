# NonlinearSolve.jl: High-Performance Unified Nonlinear Solvers

NonlinearSolve.jl is a unified interface for the nonlinear solving packages of
Julia. It includes its own high-performance nonlinear solvers which include the
ability to swap out to fast direct and iterative linear solvers, along with the
ability to use sparse automatic differentiation for Jacobian construction and
Jacobian-vector products. It interfaces with other packages of the Julia ecosystem
to make it easy to test alternative solver packages and pass small types to
control algorithm swapping. It also interfaces with the
[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) world of symbolic
modeling to allow for automatically generating high-performance code.

Performance is key: the current methods are made to be highly performant on
scalar and statically sized small problems, with options for large-scale systems.
If you run into any performance issues, please file an issue.
Consult the [NonlinearSystemSolvers](@ref nonlinearsystemsolvers) page for
information on how to import solvers from different packages.

## Installation

To install NonlinearSolve.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("NonlinearSolve")
```

## Contributing

- Please refer to the
  [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
  for guidance on PRs, issues, and other matters relating to contributing to SciML.
- See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
- There are a few community forums:
    - The #diffeq-bridged and #sciml-bridged channels in the
      [Julia Slack](https://julialang.org/slack/)
    - The #diffeq-bridged and #sciml-bridged channels in the
      [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
    - On the [Julia Discourse forums](https://discourse.julialang.org)
    - See also [SciML Community page](https://sciml.ai/community/)

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```
```@example
using Pkg # hide
Pkg.status() # hide
```
```@raw html
</details>
```
```@raw html
<details><summary>and using this machine and Julia version.</summary>
```
```@example
using InteractiveUtils # hide
versioninfo() # hide
```
```@raw html
</details>
```
```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```
```@example
using Pkg # hide
Pkg.status(;mode = PKGMODE_MANIFEST) # hide
```
```@raw html
</details>
```
```@raw html
You can also download the
<a href="
```
```@eval
using TOML
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = "https://github.com/SciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Manifest.toml"
```
```@raw html
">manifest</a> file and the
<a href="
```
```@eval
using TOML
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = "https://github.com/SciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Project.toml"
```
```@raw html
">project</a> file.
```
