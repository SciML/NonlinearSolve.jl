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
