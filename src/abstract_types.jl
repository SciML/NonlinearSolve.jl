const __internal_init = InternalAPI.init
const __internal_solve! = InternalAPI.solve!

"""
    AbstractNonlinearSolveAlgorithm{name} <: AbstractNonlinearAlgorithm

Abstract Type for all NonlinearSolve.jl Algorithms. `name` can be used to define custom
dispatches by wrapped solvers.

### Interface Functions

  - `concrete_jac(alg)`: whether or not the algorithm uses a concrete Jacobian. Defaults
    to `nothing`.
  - `get_name(alg)`: get the name of the algorithm.
"""
abstract type AbstractNonlinearSolveAlgorithm{name} <: AbstractNonlinearAlgorithm end

function Base.show(io::IO, alg::AbstractNonlinearSolveAlgorithm)
    __show_algorithm(io, alg, get_name(alg), 0)
end

get_name(::AbstractNonlinearSolveAlgorithm{name}) where {name} = name

"""
    AbstractNonlinearSolveExtensionAlgorithm <: AbstractNonlinearSolveAlgorithm{:Extension}

Abstract Type for all NonlinearSolve.jl Extension Algorithms, i.e. wrappers over 3rd party
solvers.
"""
abstract type AbstractNonlinearSolveExtensionAlgorithm <:
              AbstractNonlinearSolveAlgorithm{:Extension} end

"""
    AbstractNonlinearSolveCache{iip, timeit}

Abstract Type for all NonlinearSolve.jl Caches.

### Interface Functions

  - `get_fu(cache)`: get the residual.
  - `get_u(cache)`: get the current state.
  - `set_fu!(cache, fu)`: set the residual.
  - `set_u!(cache, u)`: set the current state.
  - `reinit!(cache, u0; kwargs...)`: reinitialize the cache with the initial state `u0` and
    any additional keyword arguments.
  - `step!(cache; kwargs...)`: See [`SciMLBase.step!`](@ref) for more details.
  - `not_terminated(cache)`: whether or not the solver has terminated.
  - `isinplace(cache)`: whether or not the solver is inplace.
"""
abstract type AbstractNonlinearSolveCache{iip, timeit} end

function SymbolicIndexingInterface.symbolic_container(cache::AbstractNonlinearSolveCache)
    return cache.prob
end
function SymbolicIndexingInterface.parameter_values(cache::AbstractNonlinearSolveCache)
    return parameter_values(symbolic_container(cache))
end
function SymbolicIndexingInterface.state_values(cache::AbstractNonlinearSolveCache)
    return state_values(symbolic_container(cache))
end

function Base.getproperty(cache::AbstractNonlinearSolveCache, sym::Symbol)
    sym == :ps && return ParameterIndexingProxy(cache)
    return getfield(cache, sym)
end

function Base.getindex(cache::AbstractNonlinearSolveCache, sym)
    return getu(cache, sym)(cache)
end

function Base.setindex!(cache::AbstractNonlinearSolveCache, val, sym)
    return setu(cache, sym)(cache, val)
end

function Base.show(io::IO, cache::AbstractNonlinearSolveCache)
    __show_cache(io, cache, 0)
end

function __show_cache(io::IO, cache::AbstractNonlinearSolveCache, indent = 0)
    println(io, "$(nameof(typeof(cache)))(")
    __show_algorithm(io, cache.alg,
        (" "^(indent + 4)) * "alg = " * string(get_name(cache.alg)), indent + 4)

    ustr = sprint(show, get_u(cache); context = (:compact => true, :limit => true))
    println(io, ",\n" * (" "^(indent + 4)) * "u = $(ustr),")

    residstr = sprint(show, get_fu(cache); context = (:compact => true, :limit => true))
    println(io, (" "^(indent + 4)) * "residual = $(residstr),")

    normstr = sprint(
        show, norm(get_fu(cache), Inf); context = (:compact => true, :limit => true))
    println(io, (" "^(indent + 4)) * "inf-norm(residual) = $(normstr),")

    println(io, " "^(indent + 4) * "nsteps = ", cache.stats.nsteps, ",")
    println(io, " "^(indent + 4) * "retcode = ", cache.retcode)
    print(io, " "^(indent) * ")")
end

SciMLBase.isinplace(::AbstractNonlinearSolveCache{iip}) where {iip} = iip

get_fu(cache::AbstractNonlinearSolveCache) = cache.fu
get_u(cache::AbstractNonlinearSolveCache) = cache.u
set_fu!(cache::AbstractNonlinearSolveCache, fu) = (cache.fu = fu)
SciMLBase.set_u!(cache::AbstractNonlinearSolveCache, u) = (cache.u = u)

function SciMLBase.reinit!(cache::AbstractNonlinearSolveCache; kwargs...)
    return reinit_cache!(cache; kwargs...)
end
function SciMLBase.reinit!(cache::AbstractNonlinearSolveCache, u0; kwargs...)
    return reinit_cache!(cache; u0, kwargs...)
end

# XXX: Move to NonlinearSolveQuasiNewton
# Approximate Jacobian Algorithms
"""
    AbstractApproximateJacobianStructure

Abstract Type for all Approximate Jacobian Structures used in NonlinearSolve.jl.

### Interface Functions

  - `stores_full_jacobian(alg)`: whether or not the algorithm stores the full Jacobian.
    Defaults to `false`.
  - `get_full_jacobian(cache, alg, J)`: get the full Jacobian. Defaults to throwing an
    error if `stores_full_jacobian(alg)` is `false`.
"""
abstract type AbstractApproximateJacobianStructure end

stores_full_jacobian(::AbstractApproximateJacobianStructure) = false
function get_full_jacobian(cache, alg::AbstractApproximateJacobianStructure, J)
    stores_full_jacobian(alg) && return J
    error("This algorithm does not store the full Jacobian. Define `get_full_jacobian` for \
           this algorithm.")
end

"""
    AbstractJacobianInitialization

Abstract Type for all Jacobian Initialization Algorithms used in NonlinearSolve.jl.

### Interface Functions

  - `jacobian_initialized_preinverted(alg)`: whether or not the Jacobian is initialized
    preinverted. Defaults to `false`.

### `__internal_init` specification

```julia
__internal_init(
    prob::AbstractNonlinearProblem, alg::AbstractJacobianInitialization, solver,
    f::F, fu, u, p; linsolve = missing, internalnorm::IN = L2_NORM, kwargs...)
```

Returns a [`NonlinearSolve.InitializedApproximateJacobianCache`](@ref).

All subtypes need to define
`(cache::InitializedApproximateJacobianCache)(alg::NewSubType, fu, u)` which reinitializes
the Jacobian in `cache.J`.
"""
abstract type AbstractJacobianInitialization end

function Base.show(io::IO, alg::AbstractJacobianInitialization)
    modifiers = String[]
    hasfield(typeof(alg), :structure) &&
        push!(modifiers, "structure = $(nameof(typeof(alg.structure)))()")
    print(io, "$(nameof(typeof(alg)))($(join(modifiers, ", ")))")
    return nothing
end

jacobian_initialized_preinverted(::AbstractJacobianInitialization) = false

"""
    AbstractApproximateJacobianUpdateRule{INV}

Abstract Type for all Approximate Jacobian Update Rules used in NonlinearSolve.jl.

### Interface Functions

  - `store_inverse_jacobian(alg)`: Return `INV`

### `__internal_init` specification

```julia
__internal_init(
    prob::AbstractNonlinearProblem, alg::AbstractApproximateJacobianUpdateRule, J, fu, u,
    du, args...; internalnorm::F = L2_NORM, kwargs...) where {F} -->
AbstractApproximateJacobianUpdateRuleCache{INV}
```
"""
abstract type AbstractApproximateJacobianUpdateRule{INV} end

store_inverse_jacobian(::AbstractApproximateJacobianUpdateRule{INV}) where {INV} = INV

"""
    AbstractApproximateJacobianUpdateRuleCache{INV}

Abstract Type for all Approximate Jacobian Update Rule Caches used in NonlinearSolve.jl.

### Interface Functions

  - `store_inverse_jacobian(alg)`: Return `INV`

### `__internal_solve!` specification

```julia
__internal_solve!(
    cache::AbstractApproximateJacobianUpdateRuleCache, J, fu, u, du; kwargs...) --> J / J⁻¹
```
"""
abstract type AbstractApproximateJacobianUpdateRuleCache{INV} end

store_inverse_jacobian(::AbstractApproximateJacobianUpdateRuleCache{INV}) where {INV} = INV

"""
    AbstractResetCondition

Condition for resetting the Jacobian in Quasi-Newton's methods.

### `__internal_init` specification

```julia
__internal_init(alg::AbstractResetCondition, J, fu, u, du, args...; kwargs...) -->
ResetCache
```

### `__internal_solve!` specification

```julia
__internal_solve!(cache::ResetCache, J, fu, u, du) --> Bool
```
"""
abstract type AbstractResetCondition end

"""
    AbstractTrustRegionMethod

Abstract Type for all Trust Region Methods used in NonlinearSolve.jl.

### `__internal_init` specification

```julia
__internal_init(
    prob::AbstractNonlinearProblem, alg::AbstractTrustRegionMethod, f::F, fu, u, p, args...;
    internalnorm::IF = L2_NORM, kwargs...) where {F, IF} --> AbstractTrustRegionMethodCache
```
"""
abstract type AbstractTrustRegionMethod end

"""
    AbstractTrustRegionMethodCache

Abstract Type for all Trust Region Method Caches used in NonlinearSolve.jl.

### Interface Functions

  - `last_step_accepted(cache)`: whether or not the last step was accepted. Defaults to
    `cache.last_step_accepted`. Should if overloaded if the field is not present.

### `__internal_solve!` specification

```julia
__internal_solve!(cache::AbstractTrustRegionMethodCache, J, fu, u, δu, descent_stats)
```

Returns `last_step_accepted`, updated `u_cache` and `fu_cache`. If the last step was
accepted then these values should be copied into the toplevel cache.
"""
abstract type AbstractTrustRegionMethodCache end

last_step_accepted(cache::AbstractTrustRegionMethodCache) = cache.last_step_accepted

"""
    AbstractNonlinearSolveTraceLevel

### Common Arguments

  - `freq`: Sets both `print_frequency` and `store_frequency` to `freq`.

### Common Keyword Arguments

  - `print_frequency`: Print the trace every `print_frequency` iterations if
    `show_trace == Val(true)`.
  - `store_frequency`: Store the trace every `store_frequency` iterations if
    `store_trace == Val(true)`.
"""
abstract type AbstractNonlinearSolveTraceLevel end
