"""
    Dogbox(; autodiff = nothing, jvp_autodiff = nothing, vjp_autodiff = nothing,
        linsolve = nothing, concrete_jac = nothing, gtol = nothing,
        initial_trust_radius = 1, max_trust_radius = Inf)

Rectangular trust-region dogleg method for box-constrained nonlinear least squares and
nonlinear equations. The step follows the dogleg path from a constrained Cauchy point
toward the Gauss–Newton step in the free variables, inside the intersection of the box
and an infinity-norm trust region.

Uses the shared Jacobian and linear-solver caches, preserving sparse and operator
representations. Rank deficiency requires a suitable linear solver and can cause
slow convergence, so prefer [`BoundedLevenbergMarquardt`](@ref) for such problems.
Bounds, `autodiff`, `gtol`, real state requirements, and root-versus-least-squares
termination follow [`TrustRegionReflective`](@ref), except that iterates can lie exactly
on a bound and the trust radius uses the infinity norm in the original coordinates.

See Voglis and Lagaris, *A Rectangular Trust Region Dogleg Approach for Unconstrained
and Bound Constrained Nonlinear Optimization*, WSEAS Applied Mathematics (2004).
"""
function Dogbox(; autodiff = nothing, jvp_autodiff = nothing, vjp_autodiff = nothing, linsolve = nothing, concrete_jac = nothing, gtol = nothing, initial_trust_radius = 1, max_trust_radius = Inf)
    return _native_bounded_algorithm(Val(:dogbox); autodiff, jvp_autodiff, vjp_autodiff, linsolve, concrete_jac, gtol, initial_trust_radius, max_trust_radius)
end

function _native_bounded_step!(::Val{:dogbox}, cache, J, x, f, g)
    free = _box_free(x, g, cache.lb, cache.ub)
    gn = _box_lsq(cache, J, f, eltype(x).(free), eltype(x).(.!free))
    cache.force_stop && return false
    lo = max.(cache.lb, x .- cache.radius)
    hi = min.(cache.ub, x .+ cache.radius)
    if all(lo .<= x + gn .<= hi)
        trial = x + gn
    else
        direction = -g .* free
        curvature = sum(abs2, J * direction)
        limit = _box_limit(x, direction, lo, hi)
        alpha = curvature > 0 ? min(-dot(g, direction) / curvature, limit) : limit
        cauchy = _dogbox_segment(x, direction, alpha, lo, hi)
        toward_gn = x + gn - cauchy
        beta = min(one(eltype(x)), _box_limit(cauchy, toward_gn, lo, hi))
        trial = _dogbox_segment(cauchy, toward_gn, beta, lo, hi)
    end
    u, fu = _box_trial(cache, trial)
    step = _box_vector(u) - x
    return _box_trust_update!(cache, u, fu, -_box_model(J, g, step), maximum(abs, step))
end

function _dogbox_segment(x, direction, alpha, lo, hi)
    trial = clamp.(x + alpha * direction, lo, hi)
    for i in eachindex(x)
        # Roundoff in hit times and trial coordinates must not leave a bound inactive.
        bound = direction[i] < 0 ? lo[i] : hi[i]
        iszero(direction[i]) && continue
        hit = (bound - x[i]) / direction[i]
        near_bound = isfinite(bound) && abs(trial[i] - bound) <= eps(max(abs(x[i]), abs(bound)))
        if alpha >= hit || isapprox(alpha, hit; rtol = 8 * eps(eltype(x)), atol = 0) || near_bound
            trial[i] = bound
        end
    end
    return trial
end
