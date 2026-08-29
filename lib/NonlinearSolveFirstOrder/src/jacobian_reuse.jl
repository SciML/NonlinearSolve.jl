"""
    JacobianReuse(; max_age::Int = 10, max_residual_ratio::Real = 1)

Reuse a Jacobian across accepted nonlinear iterations. This turns a first-order method into
an adaptive modified-Newton method: the current Jacobian is reused while the residual norm
continues to improve, subject to a maximum Jacobian age. Solvers of an unchanged concrete
linear system also reuse its factorization; damped and matrix-free systems retain their own
linear-solver update behavior.

The Jacobian is refreshed when any of these conditions holds:

  - `max_age` accepted steps have used the current Jacobian;
  - the new residual norm is not strictly less than `max_residual_ratio` times the previous
    residual norm;
  - a linear solve or globalization step fails with stale Jacobian information.

`max_age = 1` recomputes the Jacobian after every accepted step. Setting
`max_residual_ratio = Inf` selects purely periodic refreshes. The reuse state is reset by
`reinit!`; retaining a Jacobian across separate nonlinear solves requires the manual
`step!(cache; recompute_jacobian = false)` interface.

Pass `jacobian_reuse = JacobianReuse()` (or `jacobian_reuse = true`) to
[`NewtonRaphson`](@ref), [`TrustRegion`](@ref), or another first-order solver to enable the
policy. Jacobian reuse is disabled by default. A matrix-free Jacobian operator is bound to
the current iterate on every step, so there is nothing to reuse and the policy is inert.
"""
struct JacobianReuse{R <: Real}
    max_age::Int
    max_residual_ratio::R

    function JacobianReuse(max_age::Int, max_residual_ratio::R) where {R <: Real}
        max_age > 0 || throw(ArgumentError("`max_age` must be positive, got $max_age."))
        max_residual_ratio >= 0 || throw(
            ArgumentError(
                "`max_residual_ratio` must be nonnegative, got $max_residual_ratio."
            )
        )
        return new{R}(max_age, max_residual_ratio)
    end
end

function JacobianReuse(; max_age::Int = 10, max_residual_ratio::Real = 1)
    return JacobianReuse(max_age, max_residual_ratio)
end

normalize_jacobian_reuse(::Nothing) = nothing
normalize_jacobian_reuse(reuse::JacobianReuse) = reuse
normalize_jacobian_reuse(reuse::Bool) = reuse ? JacobianReuse() : nothing
function normalize_jacobian_reuse(reuse)
    throw(
        ArgumentError(
            "`jacobian_reuse` must be `nothing`, a `Bool`, or a `JacobianReuse`, got $(typeof(reuse))."
        )
    )
end

@concrete mutable struct JacobianReuseCache
    policy <: JacobianReuse
    residual_norm
    age::Int
    internalnorm
end

init_jacobian_reuse_cache(::Nothing, J, fu, internalnorm) = nothing
init_jacobian_reuse_cache(::JacobianReuse, ::StatefulJacobianOperator, fu, internalnorm) = nothing
function init_jacobian_reuse_cache(policy::JacobianReuse, J, fu, internalnorm)
    return JacobianReuseCache(policy, internalnorm(fu), 0, internalnorm)
end

reset_jacobian_reuse!(::Nothing, fu) = nothing
function reset_jacobian_reuse!(cache::JacobianReuseCache, fu)
    cache.residual_norm = cache.internalnorm(fu)
    cache.age = 0
    return nothing
end

jacobian_is_stale(::Nothing) = false
jacobian_is_stale(cache::JacobianReuseCache) = cache.age > 0

prepare_next_jacobian!(::Nothing, fu) = true
function prepare_next_jacobian!(cache::JacobianReuseCache, fu)
    (; policy) = cache
    residual_norm = cache.internalnorm(fu)
    cache.age += 1
    residual_improved = isfinite(residual_norm) && isfinite(cache.residual_norm) &&
        residual_norm < policy.max_residual_ratio * cache.residual_norm
    cache.residual_norm = residual_norm
    return !(residual_improved && cache.age < policy.max_age)
end
