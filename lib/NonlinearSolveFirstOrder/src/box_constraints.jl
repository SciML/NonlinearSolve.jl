function _box_bound(bound, fill_value, u::Number)
    T = eltype(u)
    bound === nothing && return T(fill_value)
    bound isa Number && return T(bound)
    return only(T.(bound))
end

function _box_bound(bound, fill_value, u)
    T = eltype(u)
    bound === nothing && return map(Returns(T(fill_value)), u)
    bound isa Number && return map(Returns(T(bound)), u)
    return T.(bound)
end

_box_projected_gradient(x::Number, g::Number, lb::Number, ub::Number) =
    x - clamp(x - g, lb, ub)
_box_projected_gradient(x, g, lb, ub) = _box_projected_gradient.(x, g, lb, ub)

_box_is_active(x, g, lb, ub) =
    lb == ub || (x <= lb && g > 0) || (x >= ub && g < 0)
_box_free(x, g, lb, ub) = .!_box_is_active.(x, g, lb, ub)

function _box_model_value(Jstep, step, gradient)
    return dot(step, gradient) + dot(Jstep, Jstep) / 2
end

_box_model(J, g, s) = _box_model_value(J * s, s, g)

function _box_cauchy_length(slope, curvature, limit)
    return curvature > 0 ? clamp(-slope / curvature, zero(limit), limit) : limit
end

function _box_update_radius(
        radius, ratio, stepnorm, max_radius;
        shrink_threshold = 1 // 4, expand_threshold = 3 // 4,
        shrink_radius = radius / 4, expand_factor = 2
    )
    shrunk = ratio < shrink_threshold
    if shrunk
        radius = shrink_radius
    elseif ratio > expand_threshold && stepnorm >= 0.95 * radius
        radius *= expand_factor
    end
    return min(radius, max_radius), shrunk
end
