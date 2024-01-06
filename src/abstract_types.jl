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
    pre_inverted::Val{INV} = Val(false), linsolve_kwargs = (;), abstol = nothing,
    reltol = nothing, alias_J::Bool = true, shared::Val{N} = Val(1),
    kwargs...) where {INV, N, uType, iip} --> AbstractDescentCache

SciMLBase.init(prob::NonlinearLeastSquaresProblem{uType, iip},
    alg::AbstractDescentAlgorithm, J, fu, u; pre_inverted::Val{INV} = Val(false),
    linsolve_kwargs = (;), abstol = nothing, reltol = nothing, alias_J::Bool = true,
    shared::Val{N} = Val(1), kwargs...) where {INV, N, uType, iip} --> AbstractDescentCache
```

  - `pre_inverted`: whether or not the Jacobian has been pre_inverted. Defaults to `False`.
    Note that for most algorithms except `NewtonDescent` setting it to `Val(true)` is
    generally a bad idea.
  - `linsolve_kwargs`: keyword arguments to pass to the linear solver. Defaults to `(;)`.
  - `abstol`: absolute tolerance for the linear solver. Defaults to `nothing`.
  - `reltol`: relative tolerance for the linear solver. Defaults to `nothing`.
  - `alias_J`: whether or not to alias the Jacobian. Defaults to `true`.
  - `shared`: Store multiple descent directions in the cache. Allows efficient and correct
    reuse of factorizations if needed,

Some of the algorithms also allow additional keyword arguments. See the documentation for
the specific algorithm for more information.

### `SciMLBase.solve!` specification

```julia
δu, success, intermediates = SciMLBase.solve!(cache::AbstractDescentCache, J, fu, u,
    idx::Val; skip_solve::Bool = false, kwargs...)
```

  - `J`: Jacobian or Inverse Jacobian (if `pre_inverted = Val(true)`).
  - `fu`: residual.
  - `u`: current state.
  - `idx`: index of the descent problem to solve and return. Defaults to `Val(1)`.
  - `skip_solve`: Skip the direction computation and return the previous direction.
    Defaults to `false`. This is useful for Trust Region Methods where the previous
    direction was rejected and we want to try with a modified trust region.
  - `kwargs`: keyword arguments to pass to the linear solver if there is one.

#### Returned values

  - `δu`: the descent direction.
  - `success`: Certain Descent Algorithms can reject a descent direction for example
    `GeodesicAcceleration`.
  - `intermediates`: A named tuple containing intermediates computed during the solve.
    For example, `GeodesicAcceleration` returns `NamedTuple{(:v, :a)}` containing the
    "velocity" and "acceleration" terms.

### Interface Functions

  - `supports_trust_region(alg)`: whether or not the algorithm supports trust region
    methods. Defaults to `false`.
  - `supports_line_search(alg)`: whether or not the algorithm supports line search
    methods. Defaults to `false`.

See also [`NewtonDescent`](@ref), [`Dogleg`](@ref), [`SteepestDescent`](@ref),
[`DampedNewton`](@ref).
"""
abstract type AbstractDescentAlgorithm end

supports_trust_region(::AbstractDescentAlgorithm) = false
supports_line_search(::AbstractDescentAlgorithm) = false

"""
    AbstractDescentCache

Abstract Type for all Descent Caches.

## Interface Functions

  - `get_du(cache)`: get the descent direction.
  - `get_du(cache, ::Val{N})`: get the `N`th descent direction.
  - `set_du!(cache, δu)`: set the descent direction.
  - `set_du!(cache, δu, ::Val{N})`: set the `N`th descent direction.
"""
abstract type AbstractDescentCache end

SciMLBase.get_du(cache) = cache.δu
SciMLBase.get_du(cache, ::Val{1}) = get_du(cache)
SciMLBase.get_du(cache, ::Val{N}) where {N} = cache.δus[N - 1]
set_du!(cache, δu) = (cache.δu = δu)
set_du!(cache, δu, ::Val{1}) = set_du!(cache, δu)
set_du!(cache, δu, ::Val{N}) where {N} = (cache.δus[N - 1] = δu)

"""
    AbstractNonlinearSolveLineSearchAlgorithm

Abstract Type for all Line Search Algorithms used in NonlinearSolve.jl.
"""
abstract type AbstractNonlinearSolveLineSearchAlgorithm end

abstract type AbstractNonlinearSolveLineSearchCache end

"""
    AbstractNonlinearSolveAlgorithm{name} <: AbstractNonlinearAlgorithm

Abstract Type for all NonlinearSolve.jl Algorithms. `name` can be used to define custom
dispatches by wrapped solvers.
"""
abstract type AbstractNonlinearSolveAlgorithm{name} <: AbstractNonlinearAlgorithm end

concrete_jac(::AbstractNonlinearSolveAlgorithm) = nothing

function Base.show(io::IO, alg::AbstractNonlinearSolveAlgorithm{name}) where {name}
    __show_algorithm(io, alg, name, 0)
end

get_name(::AbstractNonlinearSolveAlgorithm{name}) where {name} = name

abstract type AbstractNonlinearSolveExtensionAlgorithm <:
              AbstractNonlinearSolveAlgorithm{:Extension} end

"""
    AbstractNonlinearSolveCache{iip}

Abstract Type for all NonlinearSolve.jl Caches.
"""
abstract type AbstractNonlinearSolveCache{iip} end

SciMLBase.isinplace(::AbstractNonlinearSolveCache{iip}) where {iip} = iip

get_fu(cache::AbstractNonlinearSolveCache) = cache.fu
get_u(cache::AbstractNonlinearSolveCache) = cache.u
set_fu!(cache::AbstractNonlinearSolveCache, fu) = (cache.fu = fu)
SciMLBase.set_u!(cache::AbstractNonlinearSolveCache, u) = (cache.u = u)

"""
    AbstractLinearSolverCache <: Function

Wrapper Cache over LinearSolve.jl Caches.
"""
abstract type AbstractLinearSolverCache <: Function end

"""
    AbstractDampingFunction

Abstract Type for Damping Functions in DampedNewton.
"""
abstract type AbstractDampingFunction end

"""
    AbstractDampingFunctionCache

Abstract Type for the Caches created by AbstractDampingFunctions
"""
abstract type AbstractDampingFunctionCache end

function requires_normal_form_jacobian end
function requires_normal_form_rhs end

"""
    AbstractNonlinearSolveOperator <: SciMLBase.AbstractSciMLOperator

NonlinearSolve.jl houses a few custom operators. These will eventually be moved out but till
then this serves as the abstract type for them.
"""
abstract type AbstractNonlinearSolveOperator{T} <: SciMLBase.AbstractSciMLOperator{T} end

# Approximate Jacobian Algorithms

abstract type AbstractApproximateJacobianStructure end

stores_full_jacobian(::AbstractApproximateJacobianStructure) = false
function get_full_jacobian(cache, alg::AbstractApproximateJacobianStructure, J)
    stores_full_jacobian(alg) && return J
    error("This algorithm does not store the full Jacobian. Define `get_full_jacobian` for \
           this algorithm.")
end

abstract type AbstractJacobianInitialization end

function Base.show(io::IO, alg::AbstractJacobianInitialization)
    modifiers = String[]
    hasfield(typeof(alg), :structure) &&
        push!(modifiers, "structure = $(nameof(typeof(alg.structure)))()")
    print(io, "$(nameof(typeof(alg)))($(join(modifiers, ", ")))")
    return nothing
end

abstract type AbstractApproximateJacobianUpdateRule{INV} end

store_inverse_jacobian(::AbstractApproximateJacobianUpdateRule{INV}) where {INV} = INV

abstract type AbstractApproximateJacobianUpdateRuleCache{INV} end

store_inverse_jacobian(::AbstractApproximateJacobianUpdateRuleCache{INV}) where {INV} = INV

abstract type AbstractResetCondition end

abstract type AbstractTrustRegionMethod end

abstract type AbstractTrustRegionMethodCache end

last_step_accepted(cache::AbstractTrustRegionMethodCache) = cache.last_step_accepted

abstract type AbstractNonlinearSolveJacobianCache{iip} <: Function end

SciMLBase.isinplace(::AbstractNonlinearSolveJacobianCache{iip}) where {iip} = iip

# Default Printing
for aType in (AbstractTrustRegionMethod, AbstractNonlinearSolveLineSearchAlgorithm,
    AbstractResetCondition, AbstractApproximateJacobianUpdateRule)
    @eval function Base.show(io::IO, alg::$(aType))
        print(io, "$(nameof(typeof(alg)))()")
    end
end
