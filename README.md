# NonlinearSolve.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/NonlinearSolve/stable/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10397607.svg)](https://doi.org/10.5281/zenodo.10397607)

[![codecov](https://codecov.io/gh/SciML/NonlinearSolve.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/NonlinearSolve.jl)
[![Build Status](https://github.com/SciML/NonlinearSolve.jl/workflows/CI/badge.svg)](https://github.com/SciML/NonlinearSolve.jl/actions?query=workflow%3ACI)
[![Build status](https://badge.buildkite.com/413dc8df7d555cc14c262aba066503a9e7a42023f9cfb75a55.svg?branch=master)](https://buildkite.com/julialang/nonlinearsolve-dot-jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Fast implementations of root finding algorithms in Julia that satisfy the SciML common interface.

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/NonlinearSolve/stable/). Use the
[in-development documentation](https://docs.sciml.ai/NonlinearSolve/dev/) for the version of
the documentation which contains the unreleased features.

## High Level Examples

```julia
using NonlinearSolve, StaticArrays

f(u, p) = u .* u .- 2
u0 = @SVector[1.0, 1.0]
prob = NonlinearProblem(f, u0)
sol = solve(prob)

## Bracketing Methods

f(u, p) = u .* u .- 2.0
u0 = (1.0, 2.0) # brackets
prob = IntervalNonlinearProblem(f, u0)
sol = solve(prob)
```

## Citation

If you found this library to be useful in academic work, then please cite:

```bibtex
@article{pal2024nonlinearsolve,
  author = {Pal, Avik and Holtorf, Flemming and Larsson, Axel and Loman, Torkel and Utkarsh and Sch\"{a}fer, Frank and Qu, Qingyu and Edelman, Alan and Rackauckas, Chris},
  title = {NonlinearSolve.jl: High-Performance and Robust Solvers for Systems of Nonlinear Equations in Julia},
  year = {2025},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  issn = {0098-3500},
  url = {https://doi.org/10.1145/3779117},
  doi = {10.1145/3779117},
  abstract = {Efficiently solving nonlinear equations underpins numerous scientific and engineering disciplines, yet scaling these solutions for challenging system models remains a challenge. This paper presents NonlinearSolve.jl – a suite of high-performance open-source nonlinear equation solvers implemented natively in the Julia programming language. NonlinearSolve.jl distinguishes itself by offering a unified API that accommodates a diverse range of solver specifications alongside features such as automatic algorithm selection based on runtime analysis, support for static array kernels for improved GPU computation on smaller problems, and the utilization of sparse automatic differentiation and Jacobian-free Krylov methods for large-scale problem-solving. Through rigorous comparison with established tools such as PETSc SNES, Sundials KINSOL, and MINPACK, NonlinearSolve.jl demonstrates robustness and efficiency, achieving significant advancements in solving nonlinear equations while being implemented in a high-level programming language. The capabilities of NonlinearSolve.jl unlock new potentials in modeling and simulation across various domains, making it a valuable addition to the computational toolkit of researchers and practitioners alike.},
  note = {Just Accepted},
  journal = {ACM Trans. Math. Softw.},
  month = dec,
  keywords = {Nonlinear Systems, Root Finding, Sparsity Detection, Automatic Differentiation, JuliaLang}
}
```
