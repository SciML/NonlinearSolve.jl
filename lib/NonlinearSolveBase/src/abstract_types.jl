"""
    NonlinearSolveBase.InternalAPI

Developer extension namespace for nonlinear solver implementations.

Methods in this module are public only for NonlinearSolve.jl subpackages and downstream
solver packages that implement the NonlinearSolveBase interfaces. They are not intended as
the user-facing cache API; user code should prefer `solve`, `init`, `step!`, and SciMLBase
problem and solution objects.

### Interface Functions

  - `InternalAPI.init(args...; kwargs...)`: construct an algorithm-specific cache.
  - `InternalAPI.solve!(cache, args...; kwargs...)`: update an internal cache and return
    the algorithm-specific result.
  - `InternalAPI.step!(cache, args...; kwargs...)`: advance an iterative nonlinear solver
    cache by one step.
  - `InternalAPI.reinit!(cache, args...; kwargs...)`: reset a cache and any nested caches
    for a new solve.
  - `InternalAPI.reinit_self!(cache, args...; kwargs...)`: reset only the fields owned by
    `cache`; callers use this from generated nested-cache reset implementations.
"""
module InternalAPI

    using SciMLBase: NLStats

    function init end
    function solve! end
    function step! end

    function reinit! end
    function reinit_self! end

    function reinit!(x::Any; kwargs...)
        #@debug "`InternalAPI.reinit!` is not implemented for $(typeof(x))."
        return
    end
    function reinit_self!(x::Any; kwargs...)
        #@debug "`InternalAPI.reinit_self!` is not implemented for $(typeof(x))."
        return
    end

    function reinit!(stats::NLStats)
        stats.nf = 0
        stats.nsteps = 0
        stats.nfactors = 0
        stats.njacs = 0
        return stats.nsolve = 0
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

"""
    supports_line_search(alg)::Bool

Return whether the descent direction `alg` can be used with line-search globalization.

Descent algorithms should overload this trait when their `InternalAPI.solve!`
implementation accepts the line-search call pattern used by `GeneralizedFirstOrderAlgorithm`
and `QuasiNewtonAlgorithm`.

### Arguments

  - `alg`: An [`AbstractDescentDirection`](@ref).

### Examples

```julia
using NonlinearSolveBase

NonlinearSolveBase.supports_line_search(NewtonDescent())
```
"""
supports_line_search(::AbstractDescentDirection) = false

"""
    supports_trust_region(alg)::Bool

Return whether the descent direction `alg` can be used inside a trust-region method.

Descent algorithms should overload this trait when their `InternalAPI.solve!` method accepts
a `trust_region` keyword and reports whether the proposed step was accepted.

### Arguments

  - `alg`: An [`AbstractDescentDirection`](@ref).

### Examples

```julia
using NonlinearSolveBase

NonlinearSolveBase.supports_trust_region(Dogleg())
```
"""
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

"""
    set_du!(cache, δu)
    set_du!(cache, δu, ::Val{N})

Store the current descent direction in `cache`.

This developer hook is used by descent, quasi-Newton, and spectral-method caches to expose
their latest step through `SciMLBase.get_du`.

### Arguments

  - `cache`: An [`AbstractDescentCache`](@ref) or compatible solver cache.
  - `δu`: The descent direction to store.
  - `::Val{N}`: Optional index for caches storing multiple shared directions.
"""
set_du!(cache::AbstractDescentCache, δu) = (cache.δu = δu)
set_du!(cache::AbstractDescentCache, δu, ::Val{1}) = set_du!(cache, δu)
set_du!(cache::AbstractDescentCache, δu, ::Val{N}) where {N} = (cache.δus[N - 1] = δu)

"""
    last_step_accepted(cache::AbstractDescentCache) -> Bool

Return whether the most recent descent step was accepted.

The default reads `cache.last_step_accepted` when that field exists and returns `true`
otherwise. Trust-region and damping cache implementations should overload this hook when
acceptance is stored outside that field.

# Arguments

- `cache::AbstractDescentCache`: A descent or trust-region cache. If the cache does not
  have a `last_step_accepted` field, the default method assumes that the step was accepted.

# Returns

`true` when the most recent step was accepted and `false` when it was rejected.

# Examples

```julia
mutable struct MyDescentCache <: NonlinearSolveBase.AbstractDescentCache
    δu::Vector{Float64}
    last_step_accepted::Bool
end

cache = MyDescentCache([1.0], false)
NonlinearSolveBase.last_step_accepted(cache) # false
```
"""
function last_step_accepted(cache::AbstractDescentCache)
    hasfield(typeof(cache), :last_step_accepted) && return cache.last_step_accepted
    return true
end

"""
    preinverted_jacobian(cache::AbstractDescentCache) -> Bool

Return whether the cache stores an inverse Jacobian rather than the Jacobian itself.

The default reads the cache's `preinverted_jacobian` field and treats `missing` as `false`.
Descent cache implementations should provide that field or overload this hook.

# Arguments

- `cache::AbstractDescentCache`: A descent cache whose `preinverted_jacobian` field is a
  `Bool` or `Val{Bool}`, or a cache with a specialized method.

# Returns

`true` when the cache stores an inverse Jacobian and `false` when it stores the Jacobian.

# Examples

```julia
struct InvertedCache <: NonlinearSolveBase.AbstractDescentCache
    preinverted_jacobian::Val{true}
end

NonlinearSolveBase.preinverted_jacobian(InvertedCache(Val(true))) # true
```
"""
function preinverted_jacobian(cache::AbstractDescentCache)
    res = Utils.unwrap_val(Utils.safe_getproperty(cache, Val(:preinverted_jacobian)))
    res === missing && return false
    return res
end

"""
    normal_form(cache::AbstractDescentCache) -> Bool

Return whether the cache's linear solve uses normal-form equations.

The default reads the cache's `normal_form` field and treats `missing` as `false`.
Descent cache implementations should provide that field or overload this hook.

# Arguments

- `cache::AbstractDescentCache`: A descent cache whose `normal_form` field is a `Bool` or
  `Val{Bool}`, or a cache with a specialized method.

# Returns

`true` when the cache uses normal-form equations ``JᵀJ δu = Jᵀfu`` and `false` otherwise.

# Examples

```julia
struct NormalFormCache <: NonlinearSolveBase.AbstractDescentCache
    normal_form::Val{true}
end

NonlinearSolveBase.normal_form(NormalFormCache(Val(true))) # true
```
"""
function normal_form(cache::AbstractDescentCache)
    res = Utils.unwrap_val(Utils.safe_getproperty(cache, Val(:normal_form)))
    res === missing && return false
    return res
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

"""
    requires_normal_form_jacobian(alg) -> Bool

Return whether a damping function requires the Jacobian in normal form, ``JᵀJ``.

Every concrete [`AbstractDampingFunction`](@ref) must define this trait. It is queried
before the damping cache is initialized, so it must not depend on cache state. A damping
cache that is passed to this trait by `InternalAPI.solve!` must implement the same contract.

# Arguments

- `alg`: A damping function, or its cache when the solver queries the cache during a solve.

# Returns

`true` when the Jacobian must be supplied as ``JᵀJ`` and `false` when the ordinary Jacobian
is sufficient.
"""
function requires_normal_form_jacobian end

"""
    requires_normal_form_rhs(alg) -> Bool

Return whether a damping function requires the residual in normal form, ``Jᵀfu``.

Every concrete [`AbstractDampingFunction`](@ref) must define this trait. It is queried
before the damping cache is initialized, so it must not depend on cache state. A damping
cache that is passed to this trait by `InternalAPI.solve!` must implement the same contract.

# Arguments

- `alg`: A damping function, or its cache when the solver queries the cache during a solve.

# Returns

`true` when the residual must be supplied as ``Jᵀfu`` and `false` when the ordinary residual
is sufficient.
"""
function requires_normal_form_rhs end

"""
    returns_norm_form_damping(alg) -> Bool

Return whether the damping function returns a normal-form damping factor.

The default is `requires_normal_form_jacobian(alg) || requires_normal_form_rhs(alg)`. A
concrete damping function may overload this when its returned factor uses a different
representation.

# Arguments

- `alg`: A damping function or damping cache implementing the normal-form traits.

# Returns

`true` when the returned damping factor is in normal form and `false` otherwise.

# Examples

```julia
struct MyDamping <: NonlinearSolveBase.AbstractDampingFunction end
NonlinearSolveBase.requires_normal_form_jacobian(::MyDamping) = true
NonlinearSolveBase.requires_normal_form_rhs(::MyDamping) = false

NonlinearSolveBase.returns_norm_form_damping(MyDamping()) # true
```
"""
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
    return print(io, Utils.clean_sprint_struct(alg, indent))
end

"""
    AbstractNonlinearSolveCache <: AbstractNonlinearSolveBaseAPI

Abstract supertype for caches returned by `init(prob, alg; kwargs...)` for nonlinear
algorithms with a stepping implementation.

This is a developer-facing interface for packages that implement nonlinear solver
algorithms. It is not a replacement for the user-facing `solve` and `init` APIs. An
algorithm that does not provide a stepping implementation should use
[`NonlinearSolveNoInitCache`](@ref) instead of constructing a partial stepping cache.

# Fields

The default `CommonSolve.step!`, `CommonSolve.solve!`, and
`SymbolicIndexingInterface` methods read the following fields from a stepping cache:

- `prob::AbstractNonlinearProblem`: the problem being solved.
- `alg::AbstractNonlinearSolveAlgorithm`: the algorithm associated with the cache.
- `p`: the current parameter values.
- `u`: the current iterate, used by the default [`get_u`](@ref) method.
- `fu`: the residual at the current iterate, used by the default [`get_fu`](@ref) method.
- `nsteps::Integer`: the number of completed solver steps.
- `maxiters::Integer`: the maximum number of solver steps.
- `force_stop::Bool`: whether a caller or the solver has requested termination.
- `retcode::SciMLBase.ReturnCode.T`: the current solver status.
- `stats::SciMLBase.NLStats`: counters for function, Jacobian, factorization, and step work.
- `termination_cache`: the cache used by the termination-condition implementation.
- `trace`: the optional nonlinear solver trace.
- `timer`: the timer used by the default `step!` wrapper.
- `verbose`: the verbosity specification used by solver messages.

`maxtime` and `total_time` are also required when the cache reports a time limit through
`has_time_limit`. A cache may store any of these values elsewhere, but then it must
override every accessor or driver method that otherwise reads the default field.

# Interface

- [`get_u`](@ref): return the current iterate.
- [`get_fu`](@ref): return the current residual vector.
- [`get_nsteps`](@ref): return the number of completed steps.
- [`CommonSolve.step!`](@ref): advance the cache by one step.
- `CommonSolve.solve!(cache)`: run a stepping cache to termination and return a
  `SciMLBase.NonlinearSolution`.
- `SciMLBase.reinit!(cache, u0; kwargs...)`: reset the cache for a new initial state and
  solve options.
- [`get_abstol`](@ref) and [`get_reltol`](@ref): return the active tolerances.
- `SciMLBase.set_u!`, `set_fu!`, `SciMLBase.isinplace`, and the
  `SymbolicIndexingInterface` accessors: update or inspect the cache state.
- [`supports_deferred_residual`](@ref) and [`refresh_residual!`](@ref): coordinate an
  optional deferred residual evaluation.

# Extension Rules

- Implement `NonlinearSolveBase.InternalAPI.step!(cache::YourCache; kwargs...)`; the
  public `CommonSolve.step!` wrapper handles termination, timing, and the top-level step
  counters.
- Override [`get_u`](@ref) and [`get_fu`](@ref) when the iterate or residual is stored in a
  nested cache or another representation. These accessors must describe the same state
  that `step!` and `reinit!` operate on.
- Implement `NonlinearSolveBase.InternalAPI.reinit!` and preserve the cache's documented
  invariants when `SciMLBase.reinit!` is called.
- Return `true` from [`supports_deferred_residual`](@ref) only when deferring the residual
  cannot change termination or trace semantics, and implement [`refresh_residual!`](@ref)
  for that cache.
- Generic drivers should use the documented accessors rather than reaching into
  algorithm-specific fields. Solver packages may add internal fields without making them
  part of this interface.

# Examples

```julia
import NonlinearSolve
import NonlinearSolveBase

prob = NonlinearSolve.NonlinearProblem((u, p) -> u^2 - p, 1.0, 2.0)
cache = NonlinearSolve.init(prob, NonlinearSolve.NewtonRaphson())
NonlinearSolve.step!(cache)
u = NonlinearSolveBase.get_u(cache)
```
"""
abstract type AbstractNonlinearSolveCache <: AbstractNonlinearSolveBaseAPI end

"""
    get_u(cache::AbstractNonlinearSolveCache) -> u

Return the current iterate held by a nonlinear solver cache.

The default returns `cache.u`. Caches that keep the iterate elsewhere should overload this
hook, such as a polyalgorithm forwarding to its active subsolver or a ForwardDiff cache
forwarding to its wrapped primal cache.

# Arguments

- `cache::AbstractNonlinearSolveCache`: the cache whose current iterate is requested.

# Returns

The current iterate in the representation used by the cache's solver.

# Extension Rules

An overload must return the iterate that the cache will update on its next step. Generic
drivers should call this accessor rather than reading `cache.u` directly.

# Examples

```julia
u = NonlinearSolveBase.get_u(cache)
```
"""
get_u(cache::AbstractNonlinearSolveCache) = cache.u

"""
    get_fu(cache::AbstractNonlinearSolveCache) -> fu

Return the residual stored in a nonlinear solver cache: the most recent value of the
problem's residual function the solver evaluated (the full residual vector, not its norm,
for a `NonlinearLeastSquaresProblem`).

The default returns `cache.fu`, with the same overloading convention as [`get_u`](@ref).
Between steps this is the residual at [`get_u`](@ref), but a solver mid-step commits the
new iterate before re-evaluating there. A `postcondition` corrector runs at exactly such a
point and therefore sees the residual at the previous accepted iterate.

# Arguments

- `cache::AbstractNonlinearSolveCache`: the cache whose residual is requested.

# Returns

The full residual vector, not its norm, including for a
`NonlinearLeastSquaresProblem`.

# Extension Rules

An overload must use the same residual convention as the default and remain synchronized
with [`get_u`](@ref) at cache step boundaries. Generic drivers should call this accessor
instead of reading an algorithm-specific residual field.

# Examples

```julia
fu = NonlinearSolveBase.get_fu(cache)
```
"""
get_fu(cache::AbstractNonlinearSolveCache) = cache.fu

"""
    get_nsteps(cache::AbstractNonlinearSolveCache) -> Int

Return the number of solver iterations the cache has taken so far. This is the count
checked against `maxiters`, and it counts steps of the solver loop rather than function
or Jacobian evaluations, which are tracked separately in `cache.stats`.

# Arguments

- `cache::AbstractNonlinearSolveCache`: the cache whose step count is requested.

# Returns

The number of completed solver steps as an integer.

# Extension Rules

An overload must use the same count that controls the cache's iteration limit. Function and
Jacobian evaluations belong in `cache.stats` and must not be reported as solver steps.

# Examples

```julia
nsteps = NonlinearSolveBase.get_nsteps(cache)
```
"""
get_nsteps(cache::AbstractNonlinearSolveCache) = cache.nsteps

"""
    supports_deferred_residual(cache) -> Bool

Whether `cache` honours `step!(cache; evaluate_residual = false)`, that is, whether it can
end a step without evaluating the residual at the iterate the step landed on and leave
[`refresh_residual!`](@ref) to supply it on demand.

`false` for a cache that always evaluates, which is also the safe answer: a cache is free
to ignore `evaluate_residual = false`, and a driver that gets `false` here simply reads a
residual that is already current. A cache may only answer `true` where deferral is
unobservable — in particular where its termination condition depends on nothing but the
residual, since a deferred step reports no displacement and reaches the termination check
once per [`refresh_residual!`](@ref) rather than once per step.

# Arguments

- `cache::AbstractNonlinearSolveCache`: the cache whose deferred-residual capability is
  queried.

# Returns

`true` only when the cache supports the deferred-residual protocol; otherwise `false`.

# Extension Rules

The default is `false`. An overload returning `true` must also implement
[`refresh_residual!`](@ref) and preserve the termination and trace semantics described
above.
"""
supports_deferred_residual(::AbstractNonlinearSolveCache) = false

"""
    refresh_residual!(cache)

Settle a residual evaluation deferred by `step!(cache; evaluate_residual = false)`: evaluate
the problem's residual at [`get_u`](@ref), store it, and run the convergence check the step
would have run there, leaving `cache` in the state a plain `step!` would have left it in.
Does nothing when no evaluation is outstanding, so a driver may call it whenever it wants to
read [`get_fu`](@ref) without tracking which of its steps deferred — including on a cache
that never defers, which the default here covers. A cache that answers
[`supports_deferred_residual`](@ref) with `true` must override it.

The next `step!` settles an outstanding deferral itself, so a driver that only ever steps
again never needs to call this.

# Arguments

- `cache::AbstractNonlinearSolveCache`: the cache whose deferred residual should be settled.

# Returns

`nothing`. The cache is updated in place.

# Extension Rules

The default is a no-op for caches that never defer. A cache that returns `true` from
[`supports_deferred_residual`](@ref) must evaluate and store the residual at
[`get_u`](@ref), perform the corresponding convergence update, and make repeated calls
safe when no evaluation is outstanding.
"""
refresh_residual!(::AbstractNonlinearSolveCache) = nothing

set_fu!(cache::AbstractNonlinearSolveCache, fu) = (cache.fu = fu)
SciMLBase.set_u!(cache::AbstractNonlinearSolveCache, u) = (cache.u = u)

function has_time_limit(cache::AbstractNonlinearSolveCache)
    maxtime = Utils.safe_getproperty(cache, Val(:maxtime))
    return maxtime !== missing && maxtime !== nothing
end

function not_terminated(cache::AbstractNonlinearSolveCache)
    return (!cache.force_stop) & (cache.nsteps < cache.maxiters)
end

_prepare_reinit_parameters(p, ::Any) = p
_prepare_reinit_parameters(p, ::SciMLBase.DespecializedParameters) =
    SciMLBase.DespecializedParameters(p)

function SciMLBase.reinit!(cache::AbstractNonlinearSolveCache; p = cache.p, kwargs...)
    p = _prepare_reinit_parameters(p, cache.p)
    return InternalAPI.reinit!(cache; u = get_u(cache), p, kwargs...)
end
function SciMLBase.reinit!(cache::AbstractNonlinearSolveCache, u0; p = cache.p, kwargs...)
    p = _prepare_reinit_parameters(p, cache.p)
    return InternalAPI.reinit!(cache; u0, u = get_u(cache), p, kwargs...)
end

SciMLBase.isinplace(cache::AbstractNonlinearSolveCache) = SciMLBase.isinplace(cache.prob)

"""
    get_trace(cache::AbstractNonlinearSolveCache)

Return the trace object a solver cache records its iteration history into.

The default returns `cache.trace`. Caches that keep it elsewhere should overload this hook,
such as a polyalgorithm forwarding to its active subsolver.

# Examples

```julia
trace = NonlinearSolveBase.get_trace(cache)
```
"""
function get_trace(cache::AbstractNonlinearSolveCache)
    return cache.trace
end

"""
    get_termination_cache(cache::AbstractNonlinearSolveCache)

Return the termination-condition cache through which a solver cache reports its status.

The default returns `cache.termination_cache`. Caches that keep it elsewhere should overload
this hook, such as a polyalgorithm forwarding to its active subsolver.

# Examples

```julia
tc = NonlinearSolveBase.get_termination_cache(cache)
```
"""
function get_termination_cache(cache::AbstractNonlinearSolveCache)
    return cache.termination_cache
end

"""
    get_abstol(cache::AbstractNonlinearSolveCache) -> Real

Return the absolute tolerance currently stored in a nonlinear solver cache or problem.

The default reads the cache's `termination_cache`.

# Arguments

- `cache::AbstractNonlinearSolveCache`: the cache whose absolute tolerance is requested.

# Returns

The active absolute tolerance used by the cache's termination condition.

# Extension Rules

Override this method when the cache stores its termination state somewhere other than
`termination_cache`. The returned value must agree with the tolerance used by `step!`.

# Examples

```julia
abstol = NonlinearSolveBase.get_abstol(cache)
```
"""
function get_abstol(cache::AbstractNonlinearSolveCache)
    return get_abstol(get_termination_cache(cache))
end

"""
    get_reltol(cache::AbstractNonlinearSolveCache) -> Real

Return the relative tolerance currently stored in a nonlinear solver cache or problem.

The default reads the cache's `termination_cache`.

# Arguments

- `cache::AbstractNonlinearSolveCache`: the cache whose relative tolerance is requested.

# Returns

The active relative tolerance used by the cache's termination condition.

# Extension Rules

Override this method when the cache stores its termination state somewhere other than
`termination_cache`. The returned value must agree with the tolerance used by `step!`.

# Examples

```julia
reltol = NonlinearSolveBase.get_reltol(cache)
```
"""
function get_reltol(cache::AbstractNonlinearSolveCache)
    return get_reltol(get_termination_cache(cache))
end

## SII Interface
SII.symbolic_container(cache::AbstractNonlinearSolveCache) = cache.prob
SII.parameter_values(cache::AbstractNonlinearSolveCache) = cache.p
SII.state_values(cache::AbstractNonlinearSolveCache) = get_u(cache)

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
    println(io, lazy"$(nameof(typeof(cache)))(")
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
    return print(io, " "^(indent) * ")")
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

"""
    stores_full_jacobian(alg::AbstractApproximateJacobianStructure) -> Bool

Return whether an approximate-Jacobian structure retains the full Jacobian.

The default is `false`. A structure that retains a full Jacobian must overload this trait
and provide the corresponding [`get_full_jacobian`](@ref) behavior.

# Arguments

- `alg::AbstractApproximateJacobianStructure`: The approximate-Jacobian structure.

# Returns

`true` when the structure retains a full Jacobian and `false` otherwise.

# Examples

```julia
struct LowRankStructure <: NonlinearSolveBase.AbstractApproximateJacobianStructure end

NonlinearSolveBase.stores_full_jacobian(LowRankStructure()) # false
```
"""
stores_full_jacobian(::AbstractApproximateJacobianStructure) = false

"""
    get_full_jacobian(cache, alg::AbstractApproximateJacobianStructure, J)

Return the full Jacobian represented by an approximate-Jacobian cache.

The default returns `J` when [`stores_full_jacobian`](@ref) is true and throws otherwise.
Implementations that store the full Jacobian in a separate buffer should overload this hook.

# Arguments

- `cache`: The approximate-Jacobian cache, when the implementation stores the full matrix
  separately.
- `alg::AbstractApproximateJacobianStructure`: The structure describing the cache.
- `J`: The current Jacobian representation.

# Returns

The full Jacobian represented by the cache. The default returns `J` only when
[`stores_full_jacobian`](@ref) is `true`.

# Examples

```julia
struct FullStructure <: NonlinearSolveBase.AbstractApproximateJacobianStructure end
NonlinearSolveBase.stores_full_jacobian(::FullStructure) = true

J = [1.0 0.0; 0.0 1.0]
NonlinearSolveBase.get_full_jacobian(nothing, FullStructure(), J) == J
```
"""
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

"""
    jacobian_initialized_preinverted(alg::AbstractJacobianInitialization) -> Bool

Return whether a Jacobian initialization algorithm produces an inverse Jacobian.

The default is `false`; an initialization algorithm that constructs an inverse directly must
overload this trait so the enclosing solver interprets the cache correctly.

# Arguments

- `alg::AbstractJacobianInitialization`: The Jacobian initialization algorithm.

# Returns

`true` when the initialization algorithm returns an inverse Jacobian and `false` when it
returns an ordinary Jacobian.

# Examples

```julia
struct DirectInverse <: NonlinearSolveBase.AbstractJacobianInitialization end
NonlinearSolveBase.jacobian_initialized_preinverted(DirectInverse()) # false by default
```
"""
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

"""
    store_inverse_jacobian(rule) -> Bool

Return whether an approximate-Jacobian update rule stores an inverse Jacobian.

The default for a concrete rule reads its `store_inverse_jacobian` field. Update-rule cache
implementations delegate to the rule, so the same contract applies to both forms.

# Arguments

- `rule::AbstractApproximateJacobianUpdateRule`: The update rule whose stored Jacobian
  representation is being queried.

# Returns

`true` when the rule stores an inverse Jacobian and `false` when it stores an ordinary
Jacobian.

# Examples

```julia
struct DirectUpdate <: NonlinearSolveBase.AbstractApproximateJacobianUpdateRule
    store_inverse_jacobian::Bool
end

NonlinearSolveBase.store_inverse_jacobian(DirectUpdate(true)) # true
```
"""
function store_inverse_jacobian(rule::AbstractApproximateJacobianUpdateRule)
    return rule.store_inverse_jacobian
end

"""
    AbstractApproximateJacobianUpdateRuleCache

Abstract Type for all Approximate Jacobian Update Rule Caches used in NonlinearSolveBase.

### Interface Functions

  - `store_inverse_jacobian(cache)`: Return `store_inverse_jacobian(cache.rule)`
  - `reset_update_rule_state!(cache, fu)`: Reseed any residual the cache carries between
    iterations with `fu`.

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
    reset_update_rule_state!(cache::AbstractApproximateJacobianUpdateRuleCache, fu)

Reseed the update rule cache with `fu`, the residual at the iterate the enclosing solver
cache is being (re)initialized at, exactly as `InternalAPI.init` seeds it.

Secant-type update rules difference the current residual against the previous iterate's,
which they store across iterations. That stored residual is not reachable from
`InternalAPI.reinit_self!`, which runs on the nested caches before the enclosing cache has
evaluated the residual at the new iterate, so the enclosing cache calls this afterwards
instead. The default is a no-op, which is correct for update rules whose cache holds only
scratch buffers.
"""
reset_update_rule_state!(::AbstractApproximateJacobianUpdateRuleCache, fu) = nothing

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
    return esc(
        quote
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
        end
    )
end
