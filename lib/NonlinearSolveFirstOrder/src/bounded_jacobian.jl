_box_dense_ad(ad) = ad
_box_dense_ad(ad::ADTypes.AutoSparse) = ADTypes.dense_ad(ad)

_box_vector(x::Number) = [x]
_box_vector(x) = collect(vec(x))
_box_state(u::Number, x) = oftype(u, only(x))
_box_state(u::SArray, x) = typeof(u)(x)
_box_state(u, x) = reshape(eltype(u).(x), size(u))
_box_matrix(J::Number) = reshape([J], 1, 1)
_box_matrix(J) = J

function _box_bounds(prob, u)
    lb = _box_bound(prob.lb, -Inf, u)
    ub = _box_bound(prob.ub, Inf, u)
    length(lb) == length(u) == length(ub) ||
        throw(DimensionMismatch("Bounds must have the same length as u0."))
    return _box_vector(lb), _box_vector(ub)
end

function _box_difference_column(prob, fu, u, p, lb, ub, f0, i, stats)
    x = _box_vector(u)
    lb[i] == ub[i] && return zero(f0)
    T = promote_type(eltype(x), eltype(f0))
    h = cbrt(eps(T)) * max(one(T), abs(x[i]))
    plus, minus = min(x[i] + h, ub[i]), max(x[i] - h, lb[i])
    evaluate(x) = _box_vector(Utils.evaluate_f!!(prob, copy(fu), _box_state(u, x), p))
    # Center only when both sides have comparable spacing inside the box.
    if plus > x[i] && minus < x[i] && min(plus - x[i], x[i] - minus) >= h / 2
        xp, xm = copy(x), copy(x)
        xp[i], xm[i] = plus, minus
        fp, fm = evaluate(xp), evaluate(xm)
        stats.nf += 2
        hp, hm = plus - x[i], x[i] - minus
        return (hm / hp * (fp - f0) + hp / hm * (f0 - fm)) / (hp + hm)
    end
    xp = copy(x)
    xp[i] = plus - x[i] >= x[i] - minus ? plus : minus
    xp[i] == x[i] && return zero(f0)
    fp = evaluate(xp)
    stats.nf += 1
    return (fp - f0) / (xp[i] - x[i])
end

@concrete mutable struct BoxDifferenceCache
    prob
    fu
    p
    lb
    ub
    stats
    J
end

function (cache::BoxDifferenceCache)(u)
    f0 = _box_vector(Utils.evaluate_f!!(cache.prob, copy(cache.fu), u, cache.p))
    cache.stats.nf += 1
    if cache.J isa Number
        cache.J = only(
            _box_difference_column(
                cache.prob, cache.fu, u, cache.p, cache.lb, cache.ub, f0, 1, cache.stats
            )
        )
    else
        for i in 1:length(u)
            cache.J[:, i] = _box_difference_column(
                cache.prob, cache.fu, u, cache.p, cache.lb, cache.ub, f0, i, cache.stats
            )
        end
    end
    cache.stats.njacs += 1
    return cache.J
end

function InternalAPI.reinit!(cache::BoxDifferenceCache; p = cache.p, kwargs...)
    cache.p = p
    return nothing
end

NonlinearSolveBase.reused_jacobian(cache::BoxDifferenceCache, u) = cache.J

function _box_difference_product(prob, fu, u, p, lb, ub, stats, v, transposed)
    f0 = _box_vector(Utils.evaluate_f!!(prob, copy(fu), u, p))
    stats.nf += 1
    result = zeros(eltype(f0), transposed ? length(u) : length(fu))
    v = _box_vector(v)
    for i in 1:length(u)
        !transposed && iszero(v[i]) && continue
        column = _box_difference_column(prob, fu, u, p, lb, ub, f0, i, stats)
        if transposed
            result[i] = dot(column, v)
        else
            result .+= column .* v[i]
        end
    end
    return _box_state(transposed ? u : fu, result)
end

function _box_product_problem(prob, alg, fu, lb, ub, stats)
    f = prob.f
    for (name, ad, provided, transposed) in (
            (:jvp, alg.jvp_autodiff, SciMLBase.has_jvp(f), false),
            (:vjp, alg.vjp_autodiff, SciMLBase.has_vjp(f), true),
        )
        (provided || !(_box_dense_ad(ad) isa ADTypes.AutoFiniteDiff)) && continue
        product = if SciMLBase.isinplace(prob)
            (w, v, u, p) -> (w .= _box_difference_product(prob, fu, u, p, lb, ub, stats, v, transposed))
        else
            (v, u, p) -> _box_difference_product(prob, fu, u, p, lb, ub, stats, v, transposed)
        end
        if name === :jvp
            @set! f.jvp = product
        else
            @set! f.vjp = product
        end
    end
    return SciMLBase.remake(prob; f)
end

@concrete struct BoxShapeJacobianCache
    inner
    scalar_input
end

_box_as_array(::Val{true}, x) = [x]
_box_as_array(::Val{false}, x) = x
_box_from_array(::Val{true}, x) = only(x)
_box_from_array(::Val{false}, x) = x

(cache::BoxShapeJacobianCache)(u) = cache.inner(_box_as_array(cache.scalar_input, u))
NonlinearSolveBase.reused_jacobian(cache::BoxShapeJacobianCache, u) =
    NonlinearSolveBase.reused_jacobian(cache.inner, _box_as_array(cache.scalar_input, u))

function InternalAPI.reinit!(cache::BoxShapeJacobianCache; kwargs...)
    return InternalAPI.reinit!(cache.inner; kwargs...)
end

function _box_construct_jacobian_cache(prob, alg, fu, u, ad, stats)
    (u isa Number) == (fu isa Number) && return NonlinearSolveBase.construct_jacobian_cache(
        prob, alg, prob.f, fu, u, prob.p; stats, autodiff = ad,
        alg.linsolve, alg.jvp_autodiff, alg.vjp_autodiff
    )
    scalar_input, scalar_output = Val(u isa Number), Val(fu isa Number)
    original = prob.f
    f = if SciMLBase.isinplace(prob)
        (r, x, p) -> original(r, _box_from_array(scalar_input, x), p)
    else
        (x, p) -> _box_as_array(scalar_output, original(_box_from_array(scalar_input, x), p))
    end
    jac = if !SciMLBase.has_jac(original)
        nothing
    elseif SciMLBase.isinplace(prob)
        (J, x, p) -> original.jac(J, _box_from_array(scalar_input, x), p)
    else
        (x, p) -> reshape(
            _box_vector(original.jac(_box_from_array(scalar_input, x), p)), length(fu), length(u)
        )
    end
    input, output = _box_as_array(scalar_input, u), _box_as_array(scalar_output, fu)
    wrapped_f = original
    @set! wrapped_f.f = f
    @set! wrapped_f.jac = jac
    @set! wrapped_f.resid_prototype = output
    if !SciMLBase.isinplace(prob)
        if SciMLBase.has_jvp(original)
            @set! wrapped_f.jvp = (v, x, p) -> _box_as_array(
                scalar_output, original.jvp(_box_from_array(scalar_input, v), _box_from_array(scalar_input, x), p)
            )
        end
        if SciMLBase.has_vjp(original)
            @set! wrapped_f.vjp = (v, x, p) -> _box_as_array(
                scalar_input, original.vjp(_box_from_array(scalar_output, v), _box_from_array(scalar_input, x), p)
            )
        end
    end
    wrapped_prob = SciMLBase.remake(prob; f = wrapped_f, u0 = input)
    inner = NonlinearSolveBase.construct_jacobian_cache(
        wrapped_prob, alg, wrapped_f, output, input, prob.p; stats, autodiff = ad,
        alg.linsolve, alg.jvp_autodiff, alg.vjp_autodiff
    )
    return BoxShapeJacobianCache(inner, scalar_input)
end

_box_concrete_jacobian(alg, linsolve) =
    alg.concrete_jac === true || alg.concrete_jac === Val(true) || linsolve === nothing ||
    NonlinearSolveBase.needs_concrete_A(linsolve)

function _box_difference_cache(prob, fu, u, lb, ub, stats)
    T = promote_type(eltype(u), eltype(fu))
    prototype = prob.f.jac_prototype
    J = if u isa Number && fu isa Number
        zero(T)
    elseif prototype === nothing
        zeros(T, length(fu), length(u))
    else
        similar(prototype, T)
    end
    return BoxDifferenceCache(prob, fu, prob.p, lb, ub, stats, J)
end
