"""
    TrustRegionReflective(; autodiff = nothing, jvp_autodiff = nothing, vjp_autodiff = nothing,
        linsolve = nothing, concrete_jac = nothing, gtol = nothing,
        initial_trust_radius = 1, max_trust_radius = Inf)

Coleman–Li interior reflective trust-region method for box-constrained nonlinear least
squares and nonlinear equations. The quadratic model uses distance-to-bound scaling and
the Coleman–Li diagonal correction. Each iteration compares a strictly feasible scaled
trust-region step, a reflected direction, and a scaled gradient step.

Exact-bound initial guesses are moved into the strict interior; fixed variables remain
fixed. For least squares, `gtol` bounds the infinity norm of `u - clamp(u - J'F, lb, ub)`
and defaults to `sqrt(eps(eltype(u0)))`. For nonlinear equations, stationarity with a
nonzero residual returns `ReturnCode.Stalled`, never success; the default `gtol` is zero
to avoid stopping near a root before residual convergence. `abstol` controls the default
residual norm termination; other termination modes can be selected with `termination_condition`. Initial guesses must be feasible.

`linsolve`, `concrete_jac`, `autodiff`, `jvp_autodiff`, and `vjp_autodiff` use the same
Jacobian and LinearSolve caches as [`GaussNewton`](@ref). A sparse `jac_prototype` stays
sparse; choosing a Krylov solver constructs a Jacobian operator unless a concrete
Jacobian is requested. Analytic `jvp` and `vjp` callbacks are supported. Square-only
linear solvers use normal equations; rectangular least-squares solvers operate on the
scaled, diagonally augmented system. Rank-deficient and underdetermined problems
require a suitable linear solver, such as `SVDFactorization()` or `KrylovJL_LSMR()`.

The trust-region subproblem uses the subspace spanned by the scaled gradient and the
Gauss–Newton step. Only that one- or two-dimensional model is diagonalized. `AutoFiniteDiff`
uses bound-aware stencils for both concrete Jacobians and operator products; fixed
coordinates are never perturbed. `linsolve_kwargs` passed to `solve` or `init` are
forwarded to the linear-solver cache, including tolerances and preconditioners.

The positive `initial_trust_radius` is measured in scaled coordinates and is capped by
`max_trust_radius`. Only real floating-point states and real residuals are supported.

Based on Coleman and Li, *An Interior Trust Region Approach for Nonlinear Minimization
Subject to Bounds*, SIAM J. Optimization 6 (1996), 418–445,
[doi:10.1137/0806023](https://doi.org/10.1137/0806023).
"""
function TrustRegionReflective(;
        autodiff = nothing, jvp_autodiff = nothing, vjp_autodiff = nothing, linsolve = nothing, concrete_jac = nothing, gtol = nothing, initial_trust_radius = 1, max_trust_radius = Inf
    )
    return _native_bounded_algorithm(
        Val(:reflective); autodiff, jvp_autodiff, vjp_autodiff, linsolve, concrete_jac, gtol, initial_trust_radius, max_trust_radius
    )
end

function _reflective_scaling(x, g, lb, ub)
    v, c = ones(eltype(x), length(x)), zeros(eltype(x), length(x))
    for i in eachindex(x)
        if lb[i] == ub[i]
            v[i] = 0
        elseif g[i] < 0 && isfinite(ub[i])
            v[i], c[i] = ub[i] - x[i], -g[i]
        elseif g[i] > 0 && isfinite(lb[i])
            v[i], c[i] = x[i] - lb[i], g[i]
        end
    end
    return sqrt.(v), c
end

function _box_initial_model(::Val{:reflective}, alg, x, f, g, lb, ub)
    d, c = _reflective_scaling(x, g, lb, ub)
    return d, c + iszero.(d)
end

function _native_bounded_step!(::Val{:reflective}, cache, J, x, f, g)
    d, c = _reflective_scaling(x, g, cache.lb, cache.ub)
    h = _box_lsq(cache, J, f, d, c + iszero.(d), cache.radius)
    cache.force_stop && return false
    gh = d .* g
    product(q) = _box_vector(J * (d .* q))
    model(q) = dot(gh, q) + (sum(abs2, product(q)) + dot(c, q .^ 2)) / 2
    pg = maximum(abs, _box_projected_gradient(x, g, cache.lb, cache.ub))
    theta = min(prevfloat(one(eltype(x))), max(eltype(x)(0.995), 1 - pg))
    hit = _box_limit(x, d .* h, cache.lb, cache.ub)
    best = hit >= 1 ? h : theta * hit * h
    bestvalue = model(best)

    if hit < 1
        at_hit = hit * h
        direction = copy(h)
        point = x + d .* at_hit
        for i in eachindex(x)
            t = h[i] > 0 && d[i] > 0 ? (cache.ub[i] - x[i]) / (d[i] * h[i]) :
                (h[i] < 0 && d[i] > 0 ? (cache.lb[i] - x[i]) / (d[i] * h[i]) : Inf)
            if isapprox(t, hit; rtol = 8 * eps(eltype(x)), atol = 0)
                direction[i] = -direction[i]
            end
        end
        aa, ab = dot(direction, direction), dot(at_hit, direction)
        sphere = (sqrt(max(zero(aa), ab^2 + aa * (cache.radius^2 - dot(at_hit, at_hit)))) - ab) / aa
        limit = min(sphere, _box_limit(clamp.(point, cache.lb, cache.ub), d .* direction, cache.lb, cache.ub))
        slope = dot(gh, direction) + dot(product(at_hit), product(direction)) + dot(c .* at_hit, direction)
        curvature = sum(abs2, product(direction)) + dot(c, direction .^ 2)
        t = curvature > 0 ? clamp(-slope / curvature, zero(limit), limit) : limit
        reflected = theta * (at_hit + t * direction)
        value = model(reflected)
        if value < bestvalue
            best, bestvalue = reflected, value
        end
    end

    direction = -gh
    limit = min(
        cache.radius / LinearAlgebra.norm(direction),
        theta * _box_limit(x, d .* direction, cache.lb, cache.ub)
    )
    curvature = sum(abs2, product(direction)) + dot(c, direction .^ 2)
    t = curvature > 0 ? min(dot(gh, gh) / curvature, limit) : limit
    cauchy = t * direction
    if model(cauchy) < bestvalue
        best = cauchy
    end
    trial = _box_interior(x + d .* best, cache.lb, cache.ub)
    # Recompute the model for the representable step after strict-feasibility rounding.
    actual_h = map(trial - x, d) do si, di
        iszero(di) ? zero(si) : si / di
    end
    u, fu = _box_trial(cache, trial)
    correction = dot(c, actual_h .^ 2) / 2
    return _box_trust_update!(cache, u, fu, -model(actual_h), LinearAlgebra.norm(actual_h); correction)
end
