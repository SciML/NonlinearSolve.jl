module InternalAPI

using SciMLBase: NLStats

function init end
function solve! end
function step! end

function reinit! end
function reinit_self! end

function reinit!(x::Any; kwargs...)
    @debug "`InternalAPI.reinit!` is not implemented for $(typeof(x))."
    return
end
function reinit_self!(x::Any; kwargs...)
    @debug "`InternalAPI.reinit_self!` is not implemented for $(typeof(x))."
    return
end

function reinit!(stats::NLStats)
    stats.nf = 0
    stats.nsteps = 0
    stats.nfactors = 0
    stats.njacs = 0
    stats.nsolve = 0
end

end

abstract type AbstractNonlinearSolveBaseAPI end # Mostly used for pretty-printing

function Base.show(io::IO, ::MIME"text/plain", alg::AbstractNonlinearSolveBaseAPI)
    print(io, Utils.clean_sprint_struct(alg))
    return
end

"""
    AbstractDescentDirection

Abstract Type for all Descent Directions used in NonlinearSolveBase. Given the Jacobian
`J` and the residual `fu`, these algorithms compute the descent direction `δu`.

For non-square Jacobian problems, if we need to solve a linear solve problem, we use a
least squares solver by default, unless the provided `linsolve` can't handle non-square
matrices, in which case we use the normal form equations ``JᵀJ δu = Jᵀ fu``. Note that
this factorization is often the faster choice, but it is not as numerically stable as
the least squares solver.

### `InternalAPI.init` specification

```julia
InternalAPI.init(
    prob::AbstractNonlinearProblem, alg::AbstractDescentDirection, J, fu, u;
    pre_inverted::Val = Val(false), linsolve_kwargs = (;),
    abstol = nothing, reltol = nothing, alias_J::Bool = true,
    shared::Val = Val(1), kwargs...
)::AbstractDescentCache
```

  - `pre_inverted`: whether or not the Jacobian has been pre_inverted.
  - `linsolve_kwargs`: keyword arguments to pass to the linear solver.
  - `abstol`: absolute tolerance for the linear solver.
  - `reltol`: relative tolerance for the linear solver.
  - `alias_J`: whether or not to alias the Jacobian.
  - `shared`: Store multiple descent directions in the cache. Allows efficient and
    correct reuse of factorizations if needed.

Some of the algorithms also allow additional keyword arguments. See the documentation for
the specific algorithm for more information.

### Interface Functions

  - `supports_trust_region(alg)`: whether or not the algorithm supports trust region
    methods. Defaults to `false`.
  - `supports_line_search(alg)`: whether or not the algorithm supports line search
    methods. Defaults to `false`.

See also [`NewtonDescent`](@ref), [`Dogleg`](@ref), [`SteepestDescent`](@ref),
[`DampedNewtonDescent`](@ref).
"""
abstract type AbstractDescentDirection <: AbstractNonlinearSolveBaseAPI end

supports_line_search(::AbstractDescentDirection) = false
supports_trust_region(::AbstractDescentDirection) = false

function get_linear_solver(alg::AbstractDescentDirection)
    return Utils.safe_getproperty(alg, Val(:linsolve))
end

"""
    AbstractDescentCache

Abstract Type for all Descent Caches.

### `InternalAPI.solve!` specification

```julia
InternalAPI.solve!(
    cache::AbstractDescentCache, J, fu, u, idx::Val;
    skip_solve::Bool = false, new_jacobian::Bool = true, kwargs...
)::DescentResult
```

  - `J`: Jacobian or Inverse Jacobian (if `pre_inverted = Val(true)`).
  - `fu`: residual.
  - `u`: current state.
  - `idx`: index of the descent problem to solve and return. Defaults to `Val(1)`.
  - `skip_solve`: Skip the direction computation and return the previous direction.
    Defaults to `false`. This is useful for Trust Region Methods where the previous
    direction was rejected and we want to try with a modified trust region.
  - `new_jacobian`: Whether the Jacobian has been updated. Defaults to `true`.
  - `kwargs`: keyword arguments to pass to the linear solver if there is one.

#### Returned values

  - `descent_result`: Result in a [`DescentResult`](@ref).

### Interface Functions

  - `get_du(cache)`: get the descent direction.
  - `get_du(cache, ::Val{N})`: get the `N`th descent direction.
  - `set_du!(cache, δu)`: set the descent direction.
  - `set_du!(cache, δu, ::Val{N})`: set the `N`th descent direction.
  - `last_step_accepted(cache)`: whether or not the last step was accepted. Checks if the
    cache has a `last_step_accepted` field and returns it if it does, else returns `true`.
  - `preinverted_jacobian(cache)`: whether or not the Jacobian has been preinverted.
  - `normal_form(cache)`: whether or not the linear solver uses normal form.
"""
abstract type AbstractDescentCache <: AbstractNonlinearSolveBaseAPI end

SciMLBase.get_du(cache::AbstractDescentCache) = cache.δu
SciMLBase.get_du(cache::AbstractDescentCache, ::Val{1}) = SciMLBase.get_du(cache)
SciMLBase.get_du(cache::AbstractDescentCache, ::Val{N}) where {N} = cache.δus[N - 1]
set_du!(cache::AbstractDescentCache, δu) = (cache.δu = δu)
set_du!(cache::AbstractDescentCache, δu, ::Val{1}) = set_du!(cache, δu)
set_du!(cache::AbstractDescentCache, δu, ::Val{N}) where {N} = (cache.δus[N - 1] = δu)

function last_step_accepted(cache::AbstractDescentCache)
    hasfield(typeof(cache), :last_step_accepted) && return cache.last_step_accepted
    return true
end

for fname in (:preinverted_jacobian, :normal_form)
    @eval function $(fname)(alg::AbstractDescentCache)
        res = Utils.unwrap_val(Utils.safe_getproperty(alg, Val($(QuoteNode(fname)))))
        res === missing && return false
        return res
    end
end

"""
    AbstractDampingFunction

Abstract Type for Damping Functions in DampedNewton.

### `InternalAPI.init` specification

```julia
InternalAPI.init(
    prob::AbstractNonlinearProblem, f::AbstractDampingFunction, initial_damping,
    J, fu, u, args...;
    internalnorm = L2_NORM, kwargs...
)::AbstractDampingFunctionCache
```

Returns a [`NonlinearSolveBase.AbstractDampingFunctionCache`](@ref).
"""
abstract type AbstractDampingFunction <: AbstractNonlinearAlgorithm end

"""
    AbstractDampingFunctionCache

Abstract Type for the Caches created by AbstractDampingFunctions

### Interface Functions

  - `requires_normal_form_jacobian(alg)`: whether or not the Jacobian is needed in normal
    form. No default.
  - `requires_normal_form_rhs(alg)`: whether or not the residual is needed in normal form.
    No default.
  - `returns_norm_form_damping(alg)`: whether or not the damping function returns the
    damping factor in normal form. Defaults to
    `requires_normal_form_jacobian(alg) || requires_normal_form_rhs(alg)`.
  - `(cache::AbstractDampingFunctionCache)(::Nothing)`: returns the damping factor. The type
    of the damping factor returned from `solve!` is guaranteed to be the same as this.

### `InternalAPI.solve!` specification

```julia
InternalAPI.solve!(
    cache::AbstractDampingFunctionCache, J, fu, u, δu, descent_stats
)
```

Returns the damping factor.
"""
abstract type AbstractDampingFunctionCache <: AbstractNonlinearAlgorithm end

function requires_normal_form_jacobian end
function requires_normal_form_rhs end
function returns_norm_form_damping(f::F) where {F}
    return requires_normal_form_jacobian(f) || requires_normal_form_rhs(f)
end

"""
    AbstractNonlinearSolveAlgorithm <: AbstractNonlinearAlgorithm

Abstract Type for all NonlinearSolveBase Algorithms.

### Interface Functions

  - `concrete_jac(alg)`: whether or not the algorithm uses a concrete Jacobian. Defaults
    to `nothing`.
"""
abstract type AbstractNonlinearSolveAlgorithm <: AbstractNonlinearAlgorithm end

"""
    concrete_jac(alg::AbstractNonlinearSolveAlgorithm)::Bool

Whether the algorithm uses a concrete Jacobian.
"""
function concrete_jac(alg::AbstractNonlinearSolveAlgorithm)
    return concrete_jac(Utils.safe_getproperty(alg, Val(:concrete_jac)))
end
concrete_jac(::Missing) = false
concrete_jac(::Nothing) = false
concrete_jac(v::Bool) = v
concrete_jac(::Val{false}) = false
concrete_jac(::Val{true}) = true

function Base.show(io::IO, ::MIME"text/plain", alg::AbstractNonlinearSolveAlgorithm)
    print(io, Utils.clean_sprint_struct(alg, 0))
    return
end

function show_nonlinearsolve_algorithm(
        io::IO, alg::AbstractNonlinearSolveAlgorithm, name, indent::Int = 0
)
    print(io, name)
    print(io, Utils.clean_sprint_struct(alg, indent))
end

"""
    AbstractNonlinearSolveCache

Abstract Type for all NonlinearSolveBase Caches.

### Interface Functions

  - `get_fu(cache)`: get the residual.

  - `get_u(cache)`: get the current state.
  - `set_fu!(cache, fu)`: set the residual.
  - `has_time_limit(cache)`: whether or not the solver has a maximum time limit.
  - `not_terminated(cache)`: whether or not the solver has terminated.
  - `SciMLBase.set_u!(cache, u)`: set the current state.
  - `SciMLBase.reinit!(cache, u0; kwargs...)`: reinitialize the cache with the initial state
    `u0` and any additional keyword arguments.
  - `SciMLBase.isinplace(cache)`: whether or not the solver is inplace.
  - `CommonSolve.step!(cache; kwargs...)`: See [`CommonSolve.step!`](@ref) for more details.

Additionally implements `SymbolicIndexingInterface` interface Functions.

#### Expected Fields in Sub-Types

For the default interface implementations we expect the following fields to be present in
the cache:

  - `fu`: the residual.
  - `u`: the current state.
  - `maxiters`: the maximum number of iterations.
  - `nsteps`: the number of steps taken.
  - `force_stop`: whether or not the solver has been forced to stop.
  - `retcode`: the return code.
  - `stats`: `NLStats` object.
  - `alg`: the algorithm.
  - `maxtime`: the maximum time limit for the solver. (Optional)
  - `timer`: the timer for the solver. (Optional)
  - `total_time`: the total time taken by the solver. (Optional)
"""
abstract type AbstractNonlinearSolveCache <: AbstractNonlinearSolveBaseAPI end

get_u(cache::AbstractNonlinearSolveCache) = cache.u
get_fu(cache::AbstractNonlinearSolveCache) = cache.fu
set_fu!(cache::AbstractNonlinearSolveCache, fu) = (cache.fu = fu)
SciMLBase.set_u!(cache::AbstractNonlinearSolveCache, u) = (cache.u = u)

function has_time_limit(cache::AbstractNonlinearSolveCache)
    maxtime = Utils.safe_getproperty(cache, Val(:maxtime))
    return maxtime !== missing && maxtime !== nothing
end

function not_terminated(cache::AbstractNonlinearSolveCache)
    return !cache.force_stop && cache.nsteps < cache.maxiters
end

function SciMLBase.reinit!(cache::AbstractNonlinearSolveCache; kwargs...)
    return InternalAPI.reinit!(cache; kwargs...)
end
function SciMLBase.reinit!(cache::AbstractNonlinearSolveCache, u0; kwargs...)
    return InternalAPI.reinit!(cache; u0, kwargs...)
end

SciMLBase.isinplace(cache::AbstractNonlinearSolveCache) = SciMLBase.isinplace(cache.prob)

## SII Interface
SII.symbolic_container(cache::AbstractNonlinearSolveCache) = cache.prob
SII.parameter_values(cache::AbstractNonlinearSolveCache) = SII.parameter_values(cache.prob)
SII.state_values(cache::AbstractNonlinearSolveCache) = SII.state_values(cache.prob)

function Base.getproperty(cache::AbstractNonlinearSolveCache, sym::Symbol)
    if sym === :ps
        !hasfield(typeof(cache), :ps) && return SII.ParameterIndexingProxy(cache)
        return getfield(cache, :ps)
    end
    return getfield(cache, sym)
end

Base.getindex(cache::AbstractNonlinearSolveCache, sym) = SII.getu(cache, sym)(cache)
function Base.setindex!(cache::AbstractNonlinearSolveCache, val, sym)
    return SII.setu(cache, sym)(cache, val)
end

function Base.show(io::IO, ::MIME"text/plain", cache::AbstractNonlinearSolveCache)
    return show_nonlinearsolve_cache(io, cache)
end

function show_nonlinearsolve_cache(io::IO, cache::AbstractNonlinearSolveCache, indent = 0)
    println(io, "$(nameof(typeof(cache)))(")
    show_nonlinearsolve_algorithm(
        io,
        cache.alg,
        (" "^(indent + 4)) * "alg = ",
        indent + 4
    )

    ustr = sprint(show, get_u(cache); context = (:compact => true, :limit => true))
    println(io, ",\n" * (" "^(indent + 4)) * "u = $(ustr),")

    residstr = sprint(show, get_fu(cache); context = (:compact => true, :limit => true))
    println(io, (" "^(indent + 4)) * "residual = $(residstr),")

    normstr = sprint(
        show, norm(get_fu(cache), Inf); context = (:compact => true, :limit => true)
    )
    println(io, (" "^(indent + 4)) * "inf-norm(residual) = $(normstr),")

    println(io, " "^(indent + 4) * "nsteps = ", cache.stats.nsteps, ",")
    println(io, " "^(indent + 4) * "retcode = ", cache.retcode)
    print(io, " "^(indent) * ")")
end

"""
    AbstractLinearSolverCache

Abstract Type for all Linear Solvers used in NonlinearSolveBase. Subtypes of these are
meant to be constructured via [`construct_linear_solver`](@ref).
"""
abstract type AbstractLinearSolverCache <: AbstractNonlinearSolveBaseAPI end

"""
    AbstractJacobianCache

Abstract Type for all Jacobian Caches used in NonlinearSolveBase. Subtypes of these are
meant to be constructured via [`construct_jacobian_cache`](@ref).
"""
abstract type AbstractJacobianCache <: AbstractNonlinearSolveBaseAPI end

"""
    AbstractApproximateJacobianStructure

Abstract Type for all Approximate Jacobian Structures used in NonlinearSolve.jl.

### Interface Functions

  - `stores_full_jacobian(alg)`: whether or not the algorithm stores the full Jacobian.
    Defaults to `false`.
  - `get_full_jacobian(cache, alg, J)`: get the full Jacobian. Defaults to throwing an
    error if `stores_full_jacobian(alg)` is `false`.
"""
abstract type AbstractApproximateJacobianStructure <: AbstractNonlinearSolveBaseAPI end

stores_full_jacobian(::AbstractApproximateJacobianStructure) = false
function get_full_jacobian(cache, alg::AbstractApproximateJacobianStructure, J)
    stores_full_jacobian(alg) && return J
    error("This algorithm does not store the full Jacobian. Define `get_full_jacobian` for \
           this algorithm.")
end

"""
    AbstractJacobianInitialization

Abstract Type for all Jacobian Initialization Algorithms used in NonlinearSolveBase.

### Interface Functions

  - `jacobian_initialized_preinverted(alg)`: whether or not the Jacobian is initialized
    preinverted. Defaults to `false`.

### `InternalAPI.init` specification

```julia
InternalAPI.init(
    prob::AbstractNonlinearProblem, alg::AbstractJacobianInitialization, solver,
    f, fu, u, p;
    linsolve = missing, internalnorm::IN = L2_NORM, kwargs...
)::AbstractJacobianCache
```

All subtypes need to define
`(cache::AbstractJacobianCache)(alg::NewSubType, fu, u)` which reinitializes the Jacobian in
`cache.J`.
"""
abstract type AbstractJacobianInitialization <: AbstractNonlinearSolveBaseAPI end

jacobian_initialized_preinverted(::AbstractJacobianInitialization) = false

"""
    AbstractApproximateJacobianUpdateRule

Abstract Type for all Approximate Jacobian Update Rules used in NonlinearSolveBase.

### Interface Functions

  - `store_inverse_jacobian(alg)`: Return `alg.store_inverse_jacobian`

### `InternalAPI.init` specification

```julia
InternalAPI.init(
    prob::AbstractNonlinearProblem, alg::AbstractApproximateJacobianUpdateRule, J, fu, u,
    du, args...; internalnorm = L2_NORM, kwargs...
)::AbstractApproximateJacobianUpdateRuleCache
```
"""
abstract type AbstractApproximateJacobianUpdateRule <: AbstractNonlinearSolveBaseAPI end

function store_inverse_jacobian(rule::AbstractApproximateJacobianUpdateRule)
    return rule.store_inverse_jacobian
end

"""
    AbstractApproximateJacobianUpdateRuleCache

Abstract Type for all Approximate Jacobian Update Rule Caches used in NonlinearSolveBase.

### Interface Functions

  - `store_inverse_jacobian(cache)`: Return `store_inverse_jacobian(cache.rule)`

### `InternalAPI.solve!` specification

```julia
InternalAPI.solve!(
    cache::AbstractApproximateJacobianUpdateRuleCache, J, fu, u, du; kwargs...
) --> J / J⁻¹
```
"""
abstract type AbstractApproximateJacobianUpdateRuleCache <: AbstractNonlinearSolveBaseAPI end

function store_inverse_jacobian(cache::AbstractApproximateJacobianUpdateRuleCache)
    return store_inverse_jacobian(cache.rule)
end

"""
    AbstractResetCondition

Condition for resetting the Jacobian in Quasi-Newton's methods.

### `InternalAPI.init` specification

```julia
InternalAPI.init(
    alg::AbstractResetCondition, J, fu, u, du, args...; kwargs...
)::AbstractResetConditionCache
```
"""
abstract type AbstractResetCondition <: AbstractNonlinearSolveBaseAPI end

"""
    AbstractResetConditionCache

Abstract Type for all Reset Condition Caches used in NonlinearSolveBase.

### `InternalAPI.solve!` specification

```julia
InternalAPI.solve!(
    cache::AbstractResetConditionCache, J, fu, u, du; kwargs...
)::Bool
```
"""
abstract type AbstractResetConditionCache <: AbstractNonlinearSolveBaseAPI end

"""
    AbstractTrustRegionMethod

Abstract Type for all Trust Region Methods used in NonlinearSolveBase.

### `InternalAPI.init` specification

```julia
InternalAPI.init(
    prob::AbstractNonlinearProblem, alg::AbstractTrustRegionMethod, f, fu, u, p, args...;
    internalnorm = L2_NORM, kwargs...
)::AbstractTrustRegionMethodCache
```
"""
abstract type AbstractTrustRegionMethod <: AbstractNonlinearSolveBaseAPI end

"""
    AbstractTrustRegionMethodCache

Abstract Type for all Trust Region Method Caches used in NonlinearSolveBase.

### Interface Functions

  - `last_step_accepted(cache)`: whether or not the last step was accepted. Defaults to
    `cache.last_step_accepted`. Should if overloaded if the field is not present.

### `InternalAPI.solve!` specification

```julia
InternalAPI.solve!(
    cache::AbstractTrustRegionMethodCache, J, fu, u, δu, descent_stats; kwargs...
)
```

Returns `last_step_accepted`, updated `u_cache` and `fu_cache`. If the last step was
accepted then these values should be copied into the toplevel cache.
"""
abstract type AbstractTrustRegionMethodCache <: AbstractNonlinearSolveBaseAPI end

last_step_accepted(cache::AbstractTrustRegionMethodCache) = cache.last_step_accepted

# Additional Interface
"""
    callback_into_cache!(cache, internalcache, args...)

Define custom operations on `internalcache` tightly coupled with the calling `cache`.
`args...` contain the sequence of caches calling into `internalcache`.

This unfortunately makes code very tightly coupled and not modular. It is recommended to not
use this functionality unless it can't be avoided (like in `LevenbergMarquardt`).
"""
callback_into_cache!(cache, internalcache, args...) = nothing  # By default do nothing

# Helper functions to generate cache callbacks and resetting functions
macro internal_caches(cType, internal_cache_names...)
    callback_caches = map(internal_cache_names) do name
        return quote
            $(callback_into_cache!)(
                cache, getproperty(internalcache, $(name)), internalcache, args...
            )
        end
    end
    callbacks_self = map(internal_cache_names) do name
        return quote
            $(callback_into_cache!)(cache, getproperty(cache, $(name)))
        end
    end
    reinit_caches = map(internal_cache_names) do name
        return quote
            $(InternalAPI.reinit!)(getproperty(cache, $(name)), args...; kwargs...)
        end
    end
    return esc(quote
        function NonlinearSolveBase.callback_into_cache!(
                cache, internalcache::$(cType), args...
        )
            $(callback_caches...)
            return
        end
        function NonlinearSolveBase.callback_into_cache!(cache::$(cType))
            $(callbacks_self...)
            return
        end
        function NonlinearSolveBase.InternalAPI.reinit!(
                cache::$(cType), args...; kwargs...
        )
            $(reinit_caches...)
            $(InternalAPI.reinit_self!)(cache, args...; kwargs...)
            return
        end
    end)
end
