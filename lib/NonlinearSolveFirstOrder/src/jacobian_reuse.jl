const DEFAULT_JACOBIAN_REUSE_MAX_AGE = 10
const DEFAULT_JACOBIAN_REUSE_RESIDUAL_RATIO = 0.1

"""
    JACOBIAN_REUSE_SIZE_CUTOFF

Smallest `length(u0)` for which [`JACOBIAN_REUSE_AUTO`](@ref) enables reuse. Below it a
Jacobian is cheap relative to an extra nonlinear iteration, so the default keeps exact
Newton steps.
"""
const JACOBIAN_REUSE_SIZE_CUTOFF = 16

"""
    JACOBIAN_REUSE_AUTO

The `max_age` of a [`JacobianReuse`](@ref) that has not committed to a decision yet. It is
resolved against `length(u0)` when the solver cache is built, to
[`DEFAULT_JACOBIAN_REUSE_MAX_AGE`](@ref) at or above
[`JACOBIAN_REUSE_SIZE_CUTOFF`](@ref) and to `0` below it.
"""
const JACOBIAN_REUSE_AUTO = -1

"""
    JacobianReuse(; max_age::Int = $(DEFAULT_JACOBIAN_REUSE_MAX_AGE),
        max_residual_ratio::Real = $(DEFAULT_JACOBIAN_REUSE_RESIDUAL_RATIO))

Reuse a Jacobian across accepted nonlinear iterations. This turns a first-order method into
an adaptive modified-Newton method: the current Jacobian is reused while the residual norm
keeps contracting fast enough, subject to a maximum Jacobian age. Solvers of an unchanged
concrete linear system also reuse its factorization; damped and matrix-free systems retain
their own linear-solver update behavior.

`max_age` is the number of accepted steps a single Jacobian may serve. The Jacobian is
refreshed when any of these conditions holds:

  - it has served `max_age` accepted steps;
  - the new residual norm is not strictly less than `max_residual_ratio` times the previous
    residual norm;
  - a linear solve or globalization step fails with stale Jacobian information.

`max_age = 0` disables reuse and recovers exact Newton steps; it is how
`jacobian_reuse = false` is spelled, and `max_age = 1` means the same thing. Setting
`max_residual_ratio = Inf` selects purely periodic refreshes. The reuse state is reset by
`reinit!`; retaining a Jacobian across separate nonlinear solves requires the manual
`step!(cache; recompute_jacobian = false)` interface.

Pass `jacobian_reuse = JacobianReuse()` to [`NewtonRaphson`](@ref), [`TrustRegion`](@ref),
or another first-order solver to force the policy on, and `jacobian_reuse = false` to force
it off. The default, `jacobian_reuse = nothing`, leaves `max_age` at
[`JACOBIAN_REUSE_AUTO`](@ref) so that it resolves against `length(u0)`. A matrix-free
Jacobian operator is bound to the current iterate on every step, so there is nothing to
reuse and reuse is switched off for it.
"""
struct JacobianReuse{R <: Real}
    max_age::Int
    max_residual_ratio::R

    function JacobianReuse(max_age::Int, max_residual_ratio::R) where {R <: Real}
        max_age >= JACOBIAN_REUSE_AUTO || throw(
            ArgumentError(
                "`max_age` must be nonnegative or `JACOBIAN_REUSE_AUTO`, got $max_age."
            )
        )
        max_residual_ratio >= 0 || throw(
            ArgumentError(
                "`max_residual_ratio` must be nonnegative, got $max_residual_ratio."
            )
        )
        return new{R}(max_age, max_residual_ratio)
    end
end

function JacobianReuse(;
        max_age::Int = DEFAULT_JACOBIAN_REUSE_MAX_AGE,
        max_residual_ratio::Real = DEFAULT_JACOBIAN_REUSE_RESIDUAL_RATIO
    )
    return JacobianReuse(max_age, max_residual_ratio)
end

reuses_jacobian(policy::JacobianReuse) = policy.max_age > 1
is_automatic(policy::JacobianReuse) = policy.max_age == JACOBIAN_REUSE_AUTO
function without_jacobian_reuse(policy::JacobianReuse)
    return JacobianReuse(0, policy.max_residual_ratio)
end

# Every spelling of the keyword lands on one concrete policy type, so an algorithm carries
# the same type whether reuse is on, off, or still to be decided.
normalize_jacobian_reuse(::Nothing) = JacobianReuse(
    JACOBIAN_REUSE_AUTO, DEFAULT_JACOBIAN_REUSE_RESIDUAL_RATIO
)
normalize_jacobian_reuse(reuse::JacobianReuse) = reuse
function normalize_jacobian_reuse(reuse::Bool)
    return JacobianReuse(
        reuse ? DEFAULT_JACOBIAN_REUSE_MAX_AGE : 0, DEFAULT_JACOBIAN_REUSE_RESIDUAL_RATIO
    )
end
function normalize_jacobian_reuse(reuse)
    throw(
        ArgumentError(
            "`jacobian_reuse` must be `nothing`, a `Bool`, or a `JacobianReuse`, got $(typeof(reuse))."
        )
    )
end

# Only `max_age` varies, so a runtime size check cannot destabilize the solver cache.
function resolve_jacobian_reuse(policy::JacobianReuse, u)
    is_automatic(policy) || return policy
    max_age = length(u) >= JACOBIAN_REUSE_SIZE_CUTOFF ? DEFAULT_JACOBIAN_REUSE_MAX_AGE : 0
    return JacobianReuse(max_age, policy.max_residual_ratio)
end

# A matrix-free operator is rebound to the current iterate on every step, so there is never
# anything stale to recover from.
applicable_jacobian_reuse(policy::JacobianReuse, J) = policy
function applicable_jacobian_reuse(policy::JacobianReuse, ::StatefulJacobianOperator)
    return without_jacobian_reuse(policy)
end

@concrete mutable struct JacobianReuseCache
    policy <: JacobianReuse
    residual_norm
    age::Int
    internalnorm
end

function init_jacobian_reuse_cache(policy::JacobianReuse, J, fu, internalnorm)
    return JacobianReuseCache(
        applicable_jacobian_reuse(policy, J), internalnorm(fu), 0, internalnorm
    )
end

function reset_jacobian_reuse!(cache::JacobianReuseCache, fu)
    cache.age = 0
    reuses_jacobian(cache.policy) || return nothing
    cache.residual_norm = cache.internalnorm(fu)
    return nothing
end

function jacobian_is_stale(cache::JacobianReuseCache)
    return reuses_jacobian(cache.policy) && cache.age > 0
end

function prepare_next_jacobian!(cache::JacobianReuseCache, fu)
    (; policy) = cache
    reuses_jacobian(policy) || return true
    residual_norm = cache.internalnorm(fu)
    cache.age += 1
    residual_improved = isfinite(residual_norm) && isfinite(cache.residual_norm) &&
        residual_norm < policy.max_residual_ratio * cache.residual_norm
    cache.residual_norm = residual_norm
    return !(residual_improved && cache.age < policy.max_age)
end
