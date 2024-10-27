module InternalAPI

function init end
function solve! end
function reinit! end
function step! end

end

abstract type AbstractNonlinearSolveBaseAPI end # Mostly used for pretty-printing

function Base.show(io::IO, ::MIME"text/plain", alg::AbstractNonlinearSolveBaseAPI)
    main_name = nameof(typeof(alg))
    modifiers = String[]
    for field in fieldnames(typeof(alg))
        val = getfield(alg, field)
        Utils.is_default_value(val, field, getfield(alg, field)) && continue
        push!(modifiers, "$(field) = $(val)")
    end
    print(io, "$(main_name)($(join(modifiers, ", ")))")
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
  - `get_name(alg)`: get the name of the algorithm.
"""
abstract type AbstractNonlinearSolveAlgorithm <: AbstractNonlinearAlgorithm end

get_name(alg::AbstractNonlinearSolveAlgorithm) = Utils.safe_getproperty(alg, Val(:name))

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
  - `SciMLBase.step!(cache; kwargs...)`: See [`SciMLBase.step!`](@ref) for more details.
  - `SciMLBase.isinplace(cache)`: whether or not the solver is inplace.

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

# XXX: Implement this
# function Base.show(io::IO, cache::AbstractNonlinearSolveCache)
#     __show_cache(io, cache, 0)
# end

# function __show_cache(io::IO, cache::AbstractNonlinearSolveCache, indent = 0)
#     println(io, "$(nameof(typeof(cache)))(")
#     __show_algorithm(io, cache.alg,
#         (" "^(indent + 4)) * "alg = " * string(get_name(cache.alg)), indent + 4)

#     ustr = sprint(show, get_u(cache); context = (:compact => true, :limit => true))
#     println(io, ",\n" * (" "^(indent + 4)) * "u = $(ustr),")

#     residstr = sprint(show, get_fu(cache); context = (:compact => true, :limit => true))
#     println(io, (" "^(indent + 4)) * "residual = $(residstr),")

#     normstr = sprint(
#         show, norm(get_fu(cache), Inf); context = (:compact => true, :limit => true))
#     println(io, (" "^(indent + 4)) * "inf-norm(residual) = $(normstr),")

#     println(io, " "^(indent + 4) * "nsteps = ", cache.stats.nsteps, ",")
#     println(io, " "^(indent + 4) * "retcode = ", cache.retcode)
#     print(io, " "^(indent) * ")")
# end

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
