"""
    AbstractDescentAlgorithm

Given the Jacobian `J` and the residual `fu`, this type of algorithm computes the descent
direction `δu`.

For non-square Jacobian problems, if we need to solve a linear solve problem, we use a least
squares solver by default, unless the provided `linsolve` can't handle non-square matrices,
in which case we use the normal form equations ``JᵀJ δu = Jᵀ fu``. Note that this
factorization is often the faster choice, but it is not as numerically stable as the least
squares solver.

### `init_cache` specification

```julia
init_cache(prob::NonlinearProblem{uType, iip}, alg::AbstractDescentAlgorithm, J, fu, u;
    pre_inverted::Val{INV} = Val(false), linsolve_kwargs = (;), abstol = nothing,
    reltol = nothing, kwargs...) where {uType, iip}

init_cache(prob::NonlinearLeastSquaresProblem{uType, iip},
    alg::AbstractDescentAlgorithm, J, fu, u; pre_inverted::Val{INV} = Val(false),
    linsolve_kwargs = (;), abstol = nothing, reltol = nothing, kwargs...) where {uType, iip}
```

  - `pre_inverted`: whether or not the Jacobian has been pre_inverted. Defaults to `False`.
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

  - `J`: Jacobian or Inverse Jacobian (if `pre_inverted = Val(true)`).
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

supports_trust_region(::AbstractDescentAlgorithm) = false
supports_line_search(::AbstractDescentAlgorithm) = false

"""
    AbstractDescentCache

Abstract Type for all Descent Caches.
"""
abstract type AbstractDescentCache end

SciMLBase.get_du(cache) = cache.δu

"""
    AbstractNonlinearSolveLineSearchAlgorithm

Abstract Type for all Line Search Algorithms used in NonlinearSolve.jl.
"""
abstract type AbstractNonlinearSolveLineSearchAlgorithm end

"""
    AbstractNonlinearSolveAlgorithm{name}

Abstract Type for all NonlinearSolve.jl Algorithms. `name` can be used to define custom
dispatches by wrapped solvers.
"""
abstract type AbstractNonlinearSolveAlgorithm{name} <: AbstractNonlinearAlgorithm end

concrete_jac(::AbstractNonlinearSolveAlgorithm) = nothing

"""
    AbstractNonlinearSolveCache{iip}

Abstract Type for all NonlinearSolve.jl Caches.
"""
abstract type AbstractNonlinearSolveCache{iip} end

SciMLBase.isinplace(::AbstractNonlinearSolveCache{iip}) where {iip} = iip

import SciMLBase: set_u!
function get_u end
function get_fu end
function set_fu! end

function get_nf end
function get_njacs end
function get_nsteps end
function get_nsolve end
function get_nfactors end

"""
    AbstractLinearSolverCache

Wrapper Cache over LinearSolve.jl Caches.
"""
abstract type AbstractLinearSolverCache <: Function end