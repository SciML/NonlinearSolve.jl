# Projection mode for box bounds: the solver iterates in the original coordinates and clamps
# every committed iterate into [lb, ub], instead of running on the logit/log transform of
# `bounds_transform.jl`.

function _projection_bound(bound, fill_value, u)
    T = real(eltype(u))
    if bound === nothing
        return T(fill_value)
    elseif bound isa Number
        return T(bound)
    elseif u isa StaticArray
        return typeof(u)(T.(bound))
    else
        return T.(bound)
    end
end

"""
    projection_bounds(prob, u)

The lower and upper bounds of `prob` in the element type of `u`: a scalar for a missing or
scalar bound, otherwise an array shaped like `u`. Missing bounds become `∓Inf`.
"""
function projection_bounds(prob, u)
    return _projection_bound(prob.lb, -Inf, u), _projection_bound(prob.ub, Inf, u)
end

"""
    project_to_bounds(u, lb, ub)

Clamp `u` into `[lb, ub]` without mutating it.
"""
project_to_bounds(u::Number, lb, ub) = clamp(u, lb, ub)
project_to_bounds(u, lb, ub) = clamp.(u, lb, ub)

"""
    project_to_bounds!!(u, lb, ub)

Clamp `u` into `[lb, ub]`, in place when `u` is mutable. Use the return value.
"""
project_to_bounds!!(u::Number, lb, ub) = clamp(u, lb, ub)
function project_to_bounds!!(u, lb, ub)
    if ismutable(u)
        @. u = clamp(u, lb, ub)
        return u
    end
    return clamp.(u, lb, ub)
end

# The residual composed with the projection onto the box. A line search that evaluates
# this function never sees a point outside the box, and automatic differentiation through
# it gives the derivative of the projected path.
@concrete struct ProjectedWrapper{isinplace}
    f
    lb
    ub
end

function (w::ProjectedWrapper{false})(u, p)
    return w.f(project_to_bounds(u, w.lb, w.ub), p)
end

function (w::ProjectedWrapper{true})(resid, u, p)
    w.f(resid, project_to_bounds(u, w.lb, w.ub), p)
    return resid
end

SciMLBase.isinplace(::ProjectedWrapper{iip}) where {iip} = iip

"""
    projected_problem(prob, lb, ub)

`prob` with its residual composed with the projection onto `[lb, ub]`. Analytic
derivatives are dropped, since they are not derivatives of the composition.
"""
function projected_problem(prob, lb, ub)
    orig_f = prob.f
    raw_f = is_fw_wrapped(orig_f.f) ? get_raw_f(orig_f.f) : orig_f.f
    wrapped = ProjectedWrapper{SciMLBase.isinplace(prob)}(raw_f, lb, ub)
    new_f = @set orig_f.f = wrapped
    new_f = @set new_f.jac = nothing
    new_f = @set new_f.jvp = nothing
    new_f = @set new_f.vjp = nothing
    return remake(prob; f = new_f)
end
