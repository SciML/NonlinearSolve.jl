"""
    BoundedGaussNewton(; globalization = :linesearch, autodiff = nothing, jvp_autodiff = nothing,
        vjp_autodiff = nothing, linsolve = nothing, concrete_jac = nothing, gtol = nothing,
        initial_trust_radius = 1, max_trust_radius = Inf, max_backtracks = 40)

Active-set Gauss–Newton method in the original bounded coordinates. Variables fixed by
the box or satisfying the outward-gradient active-bound condition are excluded from the
linear least-squares solve. For square nonlinear equations this gives a feasible reduced
Newton path; a stationary point with nonzero residual is not reported as a root.

`globalization` selects `:linesearch` (projected Armijo search), `:trustregion` (spherical
trust region with a projected Cauchy safeguard), or `:trustregion_linesearch` (trust
region with a projected line-search fallback after rejection). `max_backtracks` bounds
each line search. Projected searches use the shared `LineSearch.ProjectedBackTracking`
cache. The trust radius is measured in the original coordinates.

Bounds, `autodiff`, `gtol`, real state requirements, and least-squares termination follow
[`TrustRegionReflective`](@ref), except that exact-bound iterates are permitted. The
reduced systems use the shared Jacobian and linear-solver caches, preserving sparse
and operator representations and honoring the same linear-solver and AD options.
This is a Gauss–Newton model of the residual
merit function, not an exact-Hessian constrained optimization method.
"""
function BoundedGaussNewton(;
        globalization = :linesearch, autodiff = nothing, jvp_autodiff = nothing, vjp_autodiff = nothing, linsolve = nothing, concrete_jac = nothing, gtol = nothing,
        initial_trust_radius = 1, max_trust_radius = Inf, max_backtracks = 40
    )
    globalization in (:linesearch, :trustregion, :trustregion_linesearch) ||
        throw(ArgumentError("globalization must be :linesearch, :trustregion, or :trustregion_linesearch."))
    max_backtracks > 0 || throw(ArgumentError("max_backtracks must be positive."))
    return _native_bounded_algorithm(
        Val(:gauss_newton); autodiff, jvp_autodiff, vjp_autodiff, linsolve, concrete_jac, gtol,
        initial_trust_radius, max_trust_radius, options = (; globalization, max_backtracks)
    )
end

function _native_bounded_step!(::Val{:gauss_newton}, cache, J, x, f, g)
    free = _box_free(x, g, cache.lb, cache.ub)
    kind = cache.alg.options.globalization
    radius = kind === :linesearch ? oftype(cache.radius, Inf) : cache.radius
    step = _box_lsq(cache, J, f, eltype(x).(free), eltype(x).(.!free), radius)
    cache.force_stop && return false
    if kind === :linesearch
        accepted = _box_linesearch!(cache, g, step)
    else
        step = clamp.(x + step, cache.lb, cache.ub) - x
        direction = -g .* free
        curvature = sum(abs2, J * direction)
        limit = radius / LinearAlgebra.norm(direction)
        alpha = _box_cauchy_length(dot(g, direction), curvature, limit)
        cauchy = clamp.(x + alpha * direction, cache.lb, cache.ub) - x
        if _box_model(J, g, cauchy) < _box_model(J, g, step)
            step = cauchy
        end
        u, fu = _box_trial(cache, x + step)
        step = _box_vector(u) - x
        accepted = _box_trust_update!(cache, u, fu, -_box_model(J, g, step), LinearAlgebra.norm(step))
        kind === :trustregion && return accepted
    end
    if !accepted && !cache.force_stop
        accepted = _box_linesearch!(cache, g, -g)
        if !accepted
            cache.retcode, cache.force_stop = ReturnCode.Stalled, true
        end
    end
    return accepted
end
