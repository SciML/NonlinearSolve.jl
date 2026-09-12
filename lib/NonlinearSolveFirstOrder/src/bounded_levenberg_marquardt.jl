"""
    BoundedLevenbergMarquardt(; autodiff = nothing, jvp_autodiff = nothing, vjp_autodiff = nothing,
        linsolve = nothing, concrete_jac = nothing, gtol = nothing,
        damping = 1e-3, max_backtracks = 40)

Native bound-constrained Levenberg–Marquardt method. An active-set solve minimizes the
regularized Gauss–Newton model over the box, followed by a feasible Armijo line search.
A projected-gradient line search supplies a fallback when the model step is unsuitable.
The positive `damping` initializes residual-scaled regularization and is adapted using
the agreement between actual and predicted reduction. `max_backtracks` limits each
line search.

Uses the shared Jacobian and linear-solver caches, preserving sparse and operator
representations. Bounds, linear solver and AD options, `gtol`, real state
requirements, and root-versus-least-squares termination follow [`TrustRegionReflective`](@ref),
except that iterates may lie exactly on a bound. Fixed coordinates are eliminated from
the subproblem. No coordinate transformation is used.

The constrained model and projected-gradient globalization follow the strategy of
Kanzow, Yamashita and Fukushima, *Levenberg–Marquardt methods with strong local convergence
properties for solving nonlinear equations with convex constraints*, Journal of Computational and Applied
Mathematics 172 (2004), 375–397. This implementation uses an active-set box subproblem rather
than a general convex-programming solver.
"""
function BoundedLevenbergMarquardt(;
        autodiff = nothing, jvp_autodiff = nothing, vjp_autodiff = nothing, linsolve = nothing, concrete_jac = nothing, gtol = nothing, damping = 1.0e-3, max_backtracks = 40
    )
    isfinite(damping) && damping > 0 || throw(ArgumentError("damping must be positive and finite."))
    max_backtracks > 0 || throw(ArgumentError("max_backtracks must be positive."))
    return _native_bounded_algorithm(Val(:lm); autodiff, jvp_autodiff, vjp_autodiff, linsolve, concrete_jac, gtol, options = (; damping, max_backtracks))
end

function _box_initial_model(::Val{:lm}, alg, x, f, g, lb, ub)
    scale = eltype(x).(lb .< ub)
    damping = _box_initial_damping(alg) * max(LinearAlgebra.norm(f), eps(eltype(x)))
    return scale, damping .* scale + (one.(scale) - scale)
end

function _box_constrained_lsq(cache, J, f, lo, hi, damping)
    n = length(lo)
    s = clamp.(zeros(eltype(lo), n), lo, hi)
    active = lo .== hi
    for _ in 1:(10 * n + 100)
        free = .!active
        candidate = copy(s)
        if any(free)
            scale = eltype(s).(free)
            diagonal = damping .* scale + .!free
            candidate += _box_lsq(
                cache, J, f + _box_vector(J * s), scale, diagonal;
                lower_rhs = sqrt(damping) .* s .* free
            )
            cache.force_stop && return s
        end
        direction = candidate - s
        alpha = min(one(eltype(s)), _box_limit(s, direction, lo, hi))
        s = clamp.(s + alpha * direction, lo, hi)
        if alpha < 1
            for i in eachindex(s)
                if (direction[i] < 0 && s[i] <= lo[i]) || (direction[i] > 0 && s[i] >= hi[i])
                    active[i] = true
                end
            end
            continue
        end
        gradient = _box_vector(adjoint(J) * (J * s + f)) + damping * s
        violation = map(eachindex(s)) do i
            !active[i] || lo[i] == hi[i] ? zero(eltype(s)) :
                (s[i] <= lo[i] ? max(-gradient[i], zero(eltype(s))) : max(gradient[i], zero(eltype(s))))
        end
        largest, index = findmax(violation)
        largest <= sqrt(eps(eltype(s))) * max(one(eltype(s)), maximum(abs, gradient)) && break
        active[index] = false
    end
    return s
end

function _native_bounded_step!(::Val{:lm}, cache, J, x, f, g)
    lambda = cache.damping * max(LinearAlgebra.norm(f), eps(eltype(x)))
    s = _box_constrained_lsq(cache, J, f, cache.lb - x, cache.ub - x, lambda)
    cache.force_stop && return false
    cost = sum(abs2, f) / 2
    accepted = _box_linesearch!(cache, x, g, s, cache.alg.options.max_backtracks)
    if !accepted
        accepted = _box_linesearch!(cache, x, g, -g, cache.alg.options.max_backtracks)
    end
    if accepted
        step = _box_vector(cache.u) - x
        predicted = -_box_model(J, g, step)
        ratio = predicted > 0 ? (cost - sum(abs2, cache.fu) / 2) / predicted : zero(cost)
        if ratio > 0.75
            cache.damping = max(cache.damping / 2, eps(eltype(x)))
        elseif ratio < 0.25
            cache.damping *= 2
        end
    else
        cache.retcode, cache.force_stop = ReturnCode.Stalled, true
    end
    return accepted
end
