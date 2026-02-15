# Element-wise scalar transforms between bounded and unbounded spaces.
# Each function handles 4 cases based on finiteness of lb/ub.

function _to_unbounded(u, lb, ub)
    has_lb = isfinite(lb)
    has_ub = isfinite(ub)
    if has_lb && has_ub
        return logit((u - lb) / (ub - lb))
    elseif has_lb
        return log(u - lb)
    elseif has_ub
        return log(ub - u)
    else
        return u
    end
end

function _from_unbounded(t, lb, ub)
    has_lb = isfinite(lb)
    has_ub = isfinite(ub)
    if has_lb && has_ub
        return lb + (ub - lb) * logistic(t)
    elseif has_lb
        return lb + exp(t)
    elseif has_ub
        return ub - exp(t)
    else
        return t
    end
end

# Clamp a value into the strict interior of [lb, ub] so that _to_unbounded (logit)
# doesn't receive 0 or 1, which would give ±Inf. Only applied once to u0 before
# the initial transform — it's a no-op if u0 is already in the interval.
#
# We use eps^(3/4) (~1.8e-12 for Float64) as the relative nudge factor. Plain eps
# (~2.2e-16) is so small that nudged values can round back to the boundary, while
# eps^(3/4) gives comfortable room without meaningfully changing the starting point.
function _clamp_to_bounds(u, lb, ub)
    has_lb = isfinite(lb)
    has_ub = isfinite(ub)
    eps_frac = eps(typeof(u))^(3/4)
    if has_lb && has_ub
        # Margin scales with interval width
        margin = (ub - lb) * eps_frac
        return clamp(u, lb + margin, ub - margin)
    elseif has_lb
        # max(abs(lb), 1) provides a scale factor so the nudge isn't zero when lb == 0
        return max(u, lb + eps_frac * max(abs(lb), one(lb)))
    elseif has_ub
        return min(u, ub - eps_frac * max(abs(ub), one(ub)))
    else
        return u
    end
end

# Normalize bounds: convert nothing to ±Inf vectors, broadcast scalars to match u0.
function _normalize_bound(bound, fill_value, u0)
    T = eltype(u0)

    return if isnothing(bound)
        fill(T(fill_value), size(u0))
    elseif bound isa Number
        fill(T(bound), size(u0))
    else
        T.(bound)
    end
end

function _normalize_bounds(lb, ub, u0)
    new_lb = _normalize_bound(lb, -Inf, u0)
    new_ub = _normalize_bound(ub, Inf, u0)
    return new_lb, new_ub
end

# Wrapper that contains the bounds and a cache to use for storing the
# transformed bounds.
@concrete struct BoundedWrapper{isinplace}
    f
    lb
    ub
    u_cache
end

function _transform_u(w::BoundedWrapper, u)
    tmp = w.u_cache isa FixedSizeDiffCache ? get_tmp(w.u_cache, u) : w.u_cache
    tmp .= _from_unbounded.(u, w.lb, w.ub)
    return tmp
end

function (w::BoundedWrapper{false})(u, p)
    transformed_u = _transform_u(w, u)
    return w.f(transformed_u, p)
end

function (w::BoundedWrapper{true})(resid, u, p)
    transformed_u = _transform_u(w, u)
    w.f(resid, transformed_u, p)
    return resid
end

SciMLBase.isinplace(w::BoundedWrapper{iip}) where {iip} = iip

# Wrap a problem function with bounds into a BoundedWrapper with no bounds. In a
# nutshell, we transform a parameter `p` with bounds `lb` and `ub` into an
# unbounded parameter `t` using the logistic function to map all values of `t`
# into the interval (lb, ub).
function transform_bounded_problem(prob, alg)
    lb, ub = _normalize_bounds(prob.lb, prob.ub, prob.u0)

    # Clamp u0 into the interior of the bounds so that _to_unbounded doesn't hit log(0)
    # or log(negative). We nudge by a small fraction of the interval width.
    u0_clamped = _clamp_to_bounds.(prob.u0, lb, ub)
    u0_transformed = _to_unbounded.(u0_clamped, lb, ub)

    # PreallocationTools is only supported by ForwardDiff so don't do any caching
    # if we're not using ForwardDiff.
    u_cache = if isnothing(alg) || isnothing(alg.autodiff) || alg.autodiff isa AutoForwardDiff
        FixedSizeDiffCache(prob.u0)
    else
        similar(prob.u0)
    end

    orig_f = prob.f
    wrapped = BoundedWrapper{SciMLBase.isinplace(prob)}(orig_f, lb, ub, u_cache)

    new_f = if orig_f isa NonlinearFunction
        @set orig_f.f = wrapped
    else
        wrapped
    end

    transformed_prob = remake(prob; f = new_f, u0 = u0_transformed, lb = nothing, ub = nothing)

    return transformed_prob
end
