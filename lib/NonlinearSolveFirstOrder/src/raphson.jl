"""
    NewtonRaphson(;
        concrete_jac = nothing, linsolve = nothing, linesearch = missing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        forcing = nothing,
    )

An advanced NewtonRaphson implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems.

### Keyword Arguments

  - `autodiff`: determines the backend used for the Jacobian. Defaults to `nothing` which
    means that a default is selected according to the problem specification.
  - `concrete_jac`: whether to build a concrete Jacobian. If a Krylov-subspace method is
    used, then the Jacobian will not be constructed. Defaults to `nothing`.
  - `linsolve`: the [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) solver used
    for the linear solves within the Newton method. Defaults to `nothing`, which means it
    uses the LinearSolve.jl default algorithm choice.
  - `linesearch`: the line search algorithm to use. Defaults to `missing` (no line search).
  - `vjp_autodiff`: backend for computing Vector-Jacobian products.
  - `jvp_autodiff`: backend for computing Jacobian-Vector products.
  - `forcing`: Adaptive forcing term strategy for Newton-Krylov methods. When using an
    iterative linear solver (e.g., `KrylovJL_GMRES()`), this controls how accurately the
    linear system is solved at each iteration. Use `EisenstatWalkerForcing2()` for the
    classical Eisenstat-Walker adaptive forcing strategy. Defaults to `nothing` (fixed
    tolerance from the termination condition).
"""
function NewtonRaphson(;
        concrete_jac = nothing, linsolve = nothing, linesearch = missing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        forcing = nothing,
    )
    return GeneralizedFirstOrderAlgorithm(;
        linesearch,
        descent = NewtonDescent(; linsolve),
        autodiff, vjp_autodiff, jvp_autodiff,
        concrete_jac,
        forcing,
        name = :NewtonRaphson
    )
end
