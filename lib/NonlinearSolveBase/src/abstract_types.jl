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

abstract type AbstractNonlinearSolveCache <: AbstractNonlinearSolveBaseAPI end

function get_u end
function get_fu end

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
