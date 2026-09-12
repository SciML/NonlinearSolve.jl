@concrete struct NativeBoundedAlgorithm <: AbstractNonlinearSolveAlgorithm
    method
    autodiff
    gtol
    initial_trust_radius
    max_trust_radius
    options
end

SciMLBase.allowsbounds(::NativeBoundedAlgorithm) = true

function _native_bounded_algorithm(
        method; autodiff = nothing, gtol = nothing, initial_trust_radius = 1,
        max_trust_radius = Inf, options = (;)
    )
    initial_trust_radius > 0 && isfinite(initial_trust_radius) ||
        throw(ArgumentError("initial_trust_radius must be positive and finite."))
    max_trust_radius >= initial_trust_radius ||
        throw(ArgumentError("max_trust_radius must be at least initial_trust_radius."))
    gtol === nothing || (isfinite(gtol) && gtol >= 0) ||
        throw(ArgumentError("gtol must be finite and nonnegative."))
    return NativeBoundedAlgorithm(
        method, autodiff, gtol, initial_trust_radius, max_trust_radius, options
    )
end

_box_vector(x::Number) = [x]
_box_vector(x) = collect(vec(x))
_box_state(u::Number, x) = oftype(u, only(x))
_box_state(u::SArray, x) = typeof(u)(x)
_box_state(u, x) = reshape(eltype(u).(x), size(u))
_box_matrix(J::Number) = reshape([J], 1, 1)
_box_matrix(J) = Matrix(J)

function _box_bounds(prob, u)
    lb = _bounded_tr_bound(prob.lb, -Inf, u)
    ub = _bounded_tr_bound(prob.ub, Inf, u)
    length(lb) == length(u) == length(ub) ||
        throw(DimensionMismatch("Bounds must have the same length as u0."))
    return _box_vector(lb), _box_vector(ub)
end

function _box_interior(x, lb, ub; initial = false)
    return map(x, lb, ub) do xi, lo, hi
        lo == hi && return lo
        left = isfinite(lo) ? nextfloat(lo) : lo
        right = isfinite(hi) ? prevfloat(hi) : hi
        left <= right || throw(ArgumentError("A nonfixed interval has no floating-point interior."))
        if initial
            margin = sqrt(eps(typeof(xi))) * max(one(xi), abs(xi))
            width = hi / 2 - lo / 2
            margin = min(margin, width)
            xi = clamp(xi, max(left, lo + margin), min(right, hi - margin))
        end
        return clamp(xi, left, right)
    end
end

_box_start(::Val, x, lb, ub) = x
_box_start(::Val{:reflective}, x, lb, ub) = _box_interior(x, lb, ub; initial = true)

@concrete mutable struct BoxDifferenceCache
    prob
    fu
    p
    lb
    ub
    stats
end

function (cache::BoxDifferenceCache)(u)
    x = _box_vector(u)
    f0 = _box_vector(Utils.evaluate_f!!(cache.prob, copy(cache.fu), u, cache.p))
    cache.stats.nf += 1
    T = promote_type(eltype(x), eltype(f0))
    J = zeros(T, length(f0), length(x))
    for i in eachindex(x)
        cache.lb[i] == cache.ub[i] && continue
        h = cbrt(eps(T)) * max(one(T), abs(x[i]))
        plus = min(x[i] + h, cache.ub[i])
        minus = max(x[i] - h, cache.lb[i])
        # A centered stencil is used only when both sides have comparable spacing.
        if plus > x[i] && minus < x[i] &&
                min(plus - x[i], x[i] - minus) >= h / 2
            xp, xm = copy(x), copy(x)
            xp[i], xm[i] = plus, minus
            fp = _box_vector(Utils.evaluate_f!!(cache.prob, copy(cache.fu), _box_state(u, xp), cache.p))
            fm = _box_vector(Utils.evaluate_f!!(cache.prob, copy(cache.fu), _box_state(u, xm), cache.p))
            cache.stats.nf += 2
            hp, hm = plus - x[i], x[i] - minus
            J[:, i] = (hm / hp * (fp - f0) + hp / hm * (f0 - fm)) / (hp + hm)
        else
            xp = copy(x)
            xp[i] = plus - x[i] >= x[i] - minus ? plus : minus
            xp[i] == x[i] && continue
            fp = _box_vector(Utils.evaluate_f!!(cache.prob, copy(cache.fu), _box_state(u, xp), cache.p))
            cache.stats.nf += 1
            J[:, i] = (fp - f0) / (xp[i] - x[i])
        end
    end
    cache.stats.njacs += 1
    return J
end

function InternalAPI.reinit!(cache::BoxDifferenceCache; p = cache.p, kwargs...)
    cache.p = p
    return nothing
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

function InternalAPI.reinit!(cache::BoxShapeJacobianCache; kwargs...)
    return InternalAPI.reinit!(cache.inner; kwargs...)
end

function _box_construct_jacobian_cache(prob, alg, fu, u, ad, stats)
    (u isa Number) == (fu isa Number) && return NonlinearSolveBase.construct_jacobian_cache(
        prob, alg, prob.f, fu, u, prob.p; stats, autodiff = ad
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
    wrapped_prob = SciMLBase.remake(prob; f = wrapped_f, u0 = input)
    inner = NonlinearSolveBase.construct_jacobian_cache(
        wrapped_prob, alg, wrapped_f, output, input, prob.p; stats, autodiff = ad
    )
    return BoxShapeJacobianCache(inner, scalar_input)
end

@concrete mutable struct NativeBoundedCache <: AbstractNonlinearSolveCache
    fu
    u
    u_cache
    du
    p
    prob
    alg
    lb
    ub
    jac_cache
    radius
    damping
    gtol
    stats::NLStats
    nsteps::Int
    maxiters::Int
    maxtime
    total_time::Float64
    timer
    termination_cache
    trace
    retcode::ReturnCode.T
    force_stop::Bool
    initializealg
    verbose
end

SciMLBase.get_du(cache::NativeBoundedCache) = cache.du
NonlinearSolveBase.@internal_caches NativeBoundedCache :jac_cache

function SciMLBase.__init(
        prob::AbstractNonlinearProblem, alg::NativeBoundedAlgorithm, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000, maxtime = nothing,
        termination_condition = NonlinearSolveBase.AbsNormTerminationMode(L2_NORM),
        stats = NLStats(0, 0, 0, 0, 0), verbose = NonlinearVerbosity(),
        initializealg = NonlinearSolveBase.NonlinearSolveDefaultInit(), kwargs...
    )
    _validate_native_bounds(prob, alg, prob.u0)
    u = copy(prob.u0)
    eltype(u) <: AbstractFloat || throw(ArgumentError("Native bounded methods require real floating-point states."))
    lb, ub = _box_bounds(prob, u)
    all(isfinite, u) || throw(ArgumentError("u0 must be finite."))
    u = _box_state(u, _box_start(alg.method, _box_vector(u), lb, ub))
    fu = Utils.evaluate_f(prob, u)
    stats.nf += 1
    eltype(fu) <: Real || throw(ArgumentError("Native bounded methods require real residuals."))
    ad = NonlinearSolveBase.select_jacobian_autodiff(prob, alg.autodiff)
    jac_cache = if ADTypes.dense_ad(ad) isa ADTypes.AutoFiniteDiff && !SciMLBase.has_jac(prob.f)
        BoxDifferenceCache(prob, fu, prob.p, lb, ub, stats)
    else
        _box_construct_jacobian_cache(prob, alg, fu, u, ad, stats)
    end
    T = eltype(u)
    _, _, tc = NonlinearSolveBase.init_termination_cache(
        prob, abstol, reltol, fu, u, termination_condition, Val(:regular)
    )
    du = zero(u)
    trace = NonlinearSolveBase.init_nonlinearsolve_trace(prob, alg, u, fu, nothing, du; kwargs...)
    cache = NativeBoundedCache(
        fu, u, copy(u), du, prob.p, prob, alg, lb, ub, jac_cache,
        T(alg.initial_trust_radius), one(T),
        alg.gtol === nothing ? (prob isa NonlinearLeastSquaresProblem ? sqrt(eps(T)) : zero(T)) : T(alg.gtol), stats,
        0, maxiters, maxtime, 0.0, get_timer_output(), tc, trace,
        ReturnCode.Default, false, initializealg, verbose
    )
    NonlinearSolveBase.run_initialization!(cache)
    return cache
end

function InternalAPI.reinit_self!(
        cache::NativeBoundedCache; u0 = cache.u, p = cache.p,
        maxiters = cache.maxiters, maxtime = cache.maxtime, kwargs...
    )
    _validate_native_bounds(cache.prob, cache.alg, u0)
    all(isfinite, u0) || throw(ArgumentError("u0 must be finite."))
    u0 = _box_state(u0, _box_start(cache.alg.method, _box_vector(u0), cache.lb, cache.ub))
    Utils.reinit_common!(cache, u0, p, false)
    cache.du = zero(cache.u)
    cache.radius = typeof(cache.radius)(cache.alg.initial_trust_radius)
    cache.damping = one(cache.damping)
    cache.nsteps, cache.maxiters, cache.maxtime = 0, maxiters, maxtime
    cache.total_time = 0.0
    cache.retcode, cache.force_stop = ReturnCode.Default, false
    InternalAPI.reinit!(cache.stats)
    cache.stats.nf = 1
    NonlinearSolveBase.reset!(cache.trace)
    NonlinearSolveBase.reset_timer!(cache.timer)
    SciMLBase.reinit!(cache.termination_cache, cache.fu, cache.u; kwargs...)
    return nothing
end

_box_projected_gradient(x, g, lb, ub) = x - clamp.(x - g, lb, ub)
_box_free(x, g, lb, ub) = (lb .< ub) .& .!(((x .<= lb) .& (g .> 0)) .| ((x .>= ub) .& (g .< 0)))
_box_model(J, g, s) = dot(g, s) + sum(abs2, J * s) / 2

function _box_limit(x, d, lb, ub)
    a = oftype(first(x), Inf)
    for i in eachindex(x)
        if d[i] > 0
            a = min(a, (ub[i] - x[i]) / d[i])
        elseif d[i] < 0
            a = min(a, (lb[i] - x[i]) / d[i])
        end
    end
    return max(zero(a), a)
end

function _box_trial(cache, x)
    u = _box_state(cache.u, clamp.(x, cache.lb, cache.ub))
    fu = Utils.evaluate_f!!(cache.prob, copy(cache.fu), u, cache.p)
    cache.stats.nf += 1
    return u, fu
end

function _box_accept!(cache, u, fu)
    cache.u_cache = copy(cache.u)
    cache.du = u - cache.u
    cache.u, cache.fu = u, fu
    NonlinearSolveBase.check_and_update!(cache, cache.fu, cache.u, cache.u_cache)
    return nothing
end

function _box_trust_update!(cache, u, fu, predicted, stepnorm; correction = 0)
    actual = (sum(abs2, cache.fu) - sum(abs2, fu)) / 2
    ratio = predicted > 0 && all(isfinite, fu) ? (actual - correction) / predicted : -Inf
    if ratio < 0.25
        cache.radius = min(cache.radius / 4, max(stepnorm / 4, eps(cache.radius)))
    elseif ratio > 0.75 && stepnorm >= 0.95 * cache.radius
        cache.radius = min(2 * cache.radius, cache.alg.max_trust_radius)
    end
    accepted = ratio > 1.0e-4 && actual > 0
    accepted && _box_accept!(cache, u, fu)
    if !accepted && (iszero(stepnorm) || cache.radius <= eps(eltype(cache.u)))
        cache.retcode, cache.force_stop = ReturnCode.Stalled, true
    end
    return accepted
end

function InternalAPI.step!(cache::NativeBoundedCache; kwargs...)
    NonlinearSolveBase.check_and_update!(cache, cache.fu, cache.u, cache.u)
    cache.force_stop && return nothing
    if !all(isfinite, cache.fu)
        cache.retcode, cache.force_stop = ReturnCode.Unstable, true
        return nothing
    end
    J = _box_matrix(cache.jac_cache(cache.u))
    if !all(isfinite, J)
        cache.retcode, cache.force_stop = ReturnCode.Unstable, true
        return nothing
    end
    x, f = _box_vector(cache.u), _box_vector(cache.fu)
    g = transpose(J) * f
    if maximum(abs, _box_projected_gradient(x, g, cache.lb, cache.ub)) <= cache.gtol
        cache.retcode = cache.prob isa NonlinearLeastSquaresProblem ? ReturnCode.Success : ReturnCode.Stalled
        cache.force_stop = true
        return nothing
    end
    accepted = _native_bounded_step!(cache.alg.method, cache, J, x, f, g)
    update_trace!(cache, accepted)
    return nothing
end

function _box_lsq(A, b, radius = Inf; damping = 0)
    decomp = LinearAlgebra.svd(A; full = false)
    s, rhs = decomp.S, transpose(decomp.U) * b
    cutoff = isempty(s) ? zero(eltype(A)) : eps(eltype(A)) * max(size(A)...) * maximum(s)
    function step(lambda)
        weights = map(s, rhs) do si, ri
            si <= cutoff && iszero(lambda) ? zero(ri) : -si * ri / (si^2 + lambda)
        end
        return decomp.V * weights
    end
    p = step(damping)
    LinearAlgebra.norm(p) <= radius && return p
    lo, hi = damping, max(one(eltype(A)), LinearAlgebra.norm(transpose(A) * b) / radius)
    while LinearAlgebra.norm(step(hi)) > radius
        hi *= 2
    end
    for _ in 1:80
        mid = lo / 2 + hi / 2
        if LinearAlgebra.norm(step(mid)) > radius
            lo = mid
        else
            hi = mid
        end
        hi - lo <= eps(eltype(A)) * max(one(hi), hi) && break
    end
    return step(hi)
end
