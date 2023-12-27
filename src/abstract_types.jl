"""
    AbstractDescentAlgorithm

Given the Jacobian `J` and the residual `fu`, this type of algorithm computes the descent
direction `δu`.

For non-square Jacobian problems, if we need to solve a linear solve problem, we use a least
squares solver by default, unless the provided `linsolve` can't handle non-square matrices,
in which case we use the normal form equations ``JᵀJ δu = Jᵀ fu``. Note that this
factorization is often the faster choice, but it is not as numerically stable as the least
squares solver.

### `SciMLBase.init` specification

```julia
SciMLBase.init(prob::NonlinearProblem{uType, iip}, alg::AbstractDescentAlgorithm, J, fu, u;
    preinverted::Val{INV} = Val(false), linsolve_kwargs = (;), abstol = nothing,
    reltol = nothing, kwargs...) where {uType, iip}

SciMLBase.init(prob::NonlinearLeastSquaresProblem{uType, iip},
    alg::AbstractDescentAlgorithm, J, fu, u; preinverted::Val{INV} = Val(false),
    linsolve_kwargs = (;), abstol = nothing, reltol = nothing, kwargs...) where {uType, iip}
```

  - `preinverted`: whether or not the Jacobian has been preinverted. Defaults to `False`.
    Note that for most algorithms except `NewtonDescent` setting it to `Val(true)` is
    generally a bad idea.
  - `linsolve_kwargs`: keyword arguments to pass to the linear solver. Defaults to `(;)`.
  - `abstol`: absolute tolerance for the linear solver. Defaults to `nothing`.
  - `reltol`: relative tolerance for the linear solver. Defaults to `nothing`.

Some of the algorithms also allow additional keyword arguments. See the documentation for
the specific algorithm for more information.

### `SciMLBase.solve!` specification

```julia
SciMLBase.solve!(cache::NewtonDescentCache, J, fu, args...; skip_solve::Bool = false,
    kwargs...)
```

  - `J`: Jacobian or Inverse Jacobian (if `preinverted = Val(true)`).
  - `fu`: residual.
  - `args`: Allows for more arguments to compute the descent direction. Currently no
    algorithm uses this.
  - `skip_solve`: Skip the direction computation and return the previous direction.
    Defaults to `false`. This is useful for Trust Region Methods where the previous
    direction was rejected and we want to try with a modified trust region.
  - `kwargs`: keyword arguments to pass to the linear solver if there is one.

See also [`NewtonDescent`](@ref), [`Dogleg`](@ref), [`SteepestDescent`](@ref),
[`DampedNewton`](@ref).
"""
abstract type AbstractDescentAlgorithm end