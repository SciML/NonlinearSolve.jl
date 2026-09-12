@concrete struct NativeBoundedAlgorithm <: AbstractNonlinearSolveAlgorithm
    method
    autodiff
    jvp_autodiff
    vjp_autodiff
    linsolve
    concrete_jac
    gtol
    initial_trust_radius
    max_trust_radius
    options
end

SciMLBase.allowsbounds(::NativeBoundedAlgorithm) = true

function _native_bounded_algorithm(
        method; autodiff = nothing, jvp_autodiff = nothing, vjp_autodiff = nothing,
        linsolve = nothing, concrete_jac = nothing, gtol = nothing, initial_trust_radius = 1,
        max_trust_radius = Inf, options = (;)
    )
    initial_trust_radius > 0 && isfinite(initial_trust_radius) ||
        throw(ArgumentError("initial_trust_radius must be positive and finite."))
    max_trust_radius >= initial_trust_radius ||
        throw(ArgumentError("max_trust_radius must be at least initial_trust_radius."))
    gtol === nothing || (isfinite(gtol) && gtol >= 0) ||
        throw(ArgumentError("gtol must be finite and nonnegative."))
    return NativeBoundedAlgorithm(
        method, autodiff, jvp_autodiff, vjp_autodiff, linsolve, concrete_jac,
        gtol, initial_trust_radius, max_trust_radius, options
    )
end

_box_initial_damping(alg) = haskey(alg.options, :damping) ? alg.options.damping : 1

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
    for i in 1:length(u)
        cache.J[:, i] = _box_difference_column(
            cache.prob, cache.fu, u, cache.p, cache.lb, cache.ub, f0, i, cache.stats
        )
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
    linear_cache
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
NonlinearSolveBase.@internal_caches NativeBoundedCache :jac_cache :linear_cache
NonlinearSolveBase.get_linear_cache(cache::NativeBoundedCache) =
    NonlinearSolveBase.get_linear_cache(cache.linear_cache)

function SciMLBase.__init(
        prob::AbstractNonlinearProblem, alg::NativeBoundedAlgorithm, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000, maxtime = nothing,
        termination_condition = NonlinearSolveBase.AbsNormTerminationMode(L2_NORM),
        stats = NLStats(0, 0, 0, 0, 0), verbose = NonlinearVerbosity(),
        initializealg = NonlinearSolveBase.NonlinearSolveDefaultInit(), linsolve_kwargs = (;), kwargs...
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
    @set! alg.autodiff = ad
    finite_difference = _box_dense_ad(ad) isa ADTypes.AutoFiniteDiff
    @set! alg.jvp_autodiff = NonlinearSolveBase.select_forward_mode_autodiff(
        prob, alg.jvp_autodiff === nothing && ADTypes.mode(ad) isa Union{ADTypes.ForwardMode, ADTypes.ForwardOrReverseMode} ? ad : alg.jvp_autodiff
    )
    @set! alg.vjp_autodiff = NonlinearSolveBase.select_reverse_mode_autodiff(
        prob, alg.vjp_autodiff === nothing && (finite_difference || ADTypes.mode(ad) isa Union{ADTypes.ReverseMode, ADTypes.ForwardOrReverseMode}) ? ad : alg.vjp_autodiff
    )
    concrete = alg.concrete_jac === true || alg.concrete_jac === Val(true) || alg.linsolve === nothing ||
        NonlinearSolveBase.needs_concrete_A(alg.linsolve)
    T = eltype(u)
    jac_cache = if concrete && finite_difference &&
            !SciMLBase.has_jac(prob.f) && !(prob.f.jac_prototype isa SciMLOperators.AbstractSciMLOperator)
        prototype = prob.f.jac_prototype
        J = prototype === nothing ? zeros(T, length(fu), length(u)) :
            similar(prototype, T)
        BoxDifferenceCache(prob, fu, prob.p, lb, ub, stats, J)
    else
        ad_prob = concrete ? prob : _box_product_problem(prob, alg, fu, lb, ub, stats)
        _box_construct_jacobian_cache(ad_prob, alg, fu, u, ad, stats)
    end
    _, _, tc = NonlinearSolveBase.init_termination_cache(
        prob, abstol, reltol, fu, u, termination_condition, Val(:regular)
    )
    J = _box_matrix(jac_cache(u))
    linear_cache = _box_init_linear_cache(
        alg, J, fu, u, prob.p, lb, ub, stats,
        merge((; abstol = zero(T), reltol = eps(T)^(3 / 4), verbose = verbose.linear_verbosity), linsolve_kwargs)
    )
    du = zero(u)
    trace = NonlinearSolveBase.init_nonlinearsolve_trace(prob, alg, u, fu, nothing, du; kwargs...)
    cache = NativeBoundedCache(
        fu, u, copy(u), du, prob.p, prob, alg, lb, ub, jac_cache, linear_cache,
        T(alg.initial_trust_radius), T(_box_initial_damping(alg)),
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
    cache.damping = typeof(cache.damping)(_box_initial_damping(cache.alg))
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
    if J isa AbstractArray && !all(isfinite, J)
        cache.retcode, cache.force_stop = ReturnCode.Unstable, true
        return nothing
    end
    x, f = _box_vector(cache.u), _box_vector(cache.fu)
    g = _box_vector(adjoint(J) * f)
    if !all(isfinite, g)
        cache.retcode, cache.force_stop = ReturnCode.Unstable, true
        return nothing
    end
    if maximum(abs, _box_projected_gradient(x, g, cache.lb, cache.ub)) <= cache.gtol
        cache.retcode = cache.prob isa NonlinearLeastSquaresProblem ? ReturnCode.Success : ReturnCode.Stalled
        cache.force_stop = true
        return nothing
    end
    accepted = _native_bounded_step!(cache.alg.method, cache, J, x, f, g)
    update_trace!(cache, accepted)
    return nothing
end

@concrete mutable struct BoxLinearModel
    J
    scale
    diagonal
end

function _box_system(model, normal_form, concrete)
    J, d, c = model.J, model.scale, model.diagonal
    if concrete && J isa SciMLOperators.AbstractSciMLOperator
        J = convert(AbstractMatrix, J)
    end
    if J isa AbstractMatrix
        A = J * Diagonal(d)
        return normal_form ? transpose(A) * A + Diagonal(c) :
            vcat(A, Diagonal(sqrt.(c)))
    end
    n, m = length(d), size(J, 1)
    if normal_form
        op = (v, u, p, t) -> model.scale .* _box_vector(
            adjoint(model.J) * _box_vector(model.J * (model.scale .* v))
        ) + model.diagonal .* v
        return SciMLOperators.FunctionOperator(
            op, zeros(eltype(d), n); op_adjoint = op, islinear = true,
            issymmetric = true, ishermitian = true
        )
    end
    op = (v, u, p, t) -> vcat(
        _box_vector(model.J * (model.scale .* v)), sqrt.(model.diagonal) .* v
    )
    adjoint_op = (v, u, p, t) -> model.scale .* _box_vector(
        adjoint(model.J) * view(v, 1:m)
    ) + sqrt.(model.diagonal) .* view(v, (m + 1):(m + n))
    return SciMLOperators.FunctionOperator(
        op, zeros(eltype(d), n), zeros(eltype(d), m + n);
        op_adjoint = adjoint_op, islinear = true
    )
end

@concrete mutable struct BoxLinearCache
    model
    system
    lincache
    normal_form::Bool
    concrete::Bool
end

function InternalAPI.reinit!(cache::BoxLinearCache; u = missing, u0 = u, kwargs...)
    return InternalAPI.reinit!(cache.lincache; u = u0 === missing ? missing : _box_vector(u0), kwargs...)
end

NonlinearSolveBase.get_linear_cache(cache::BoxLinearCache) =
    NonlinearSolveBase.get_linear_cache(cache.lincache)

function _box_initial_model(::Val, alg, x, f, g, lb, ub)
    scale = eltype(x).(_box_free(x, g, lb, ub))
    return scale, one.(scale) - scale
end

function _box_init_linear_cache(alg, J, f, u, p, lb, ub, stats, linsolve_kwargs)
    x = _box_vector(u)
    f = _box_vector(f)
    g = _box_vector(adjoint(J) * f)
    scale, diagonal = _box_initial_model(alg.method, alg, x, f, g, lb, ub)
    model = BoxLinearModel(J, scale, diagonal)
    normal_form = NonlinearSolveBase.needs_square_A(alg.linsolve, x)
    concrete = alg.concrete_jac === true || alg.concrete_jac === Val(true) || alg.linsolve === nothing ||
        NonlinearSolveBase.needs_concrete_A(alg.linsolve)
    A = _box_system(model, normal_form, concrete)
    b = zeros(eltype(x), size(A, 1))
    lincache = NonlinearSolveBase.construct_linear_solver(
        alg, alg.linsolve, A, b, zero(x), p; stats, linsolve_kwargs...
    )
    return BoxLinearCache(model, A, lincache, normal_form, concrete)
end

function _box_linear_solve!(cache, J, b, scale, diagonal, lower_rhs)
    linear = cache.linear_cache
    linear.model.J = J
    linear.model.scale .= scale
    linear.model.diagonal .= diagonal
    if J isa AbstractMatrix || linear.concrete
        linear.system = _box_system(linear.model, linear.normal_form, linear.concrete)
    end
    rhs = linear.normal_form ?
        -(scale .* _box_vector(adjoint(J) * b) + sqrt.(diagonal) .* lower_rhs) :
        -vcat(b, lower_rhs)
    InternalAPI.reinit!(linear.lincache; u = _box_vector(cache.u), p = cache.p)
    result = linear.lincache(; A = linear.system, b = rhs, linu = zero(scale))
    if !result.success || !all(isfinite, result.u)
        cache.retcode, cache.force_stop = ReturnCode.InternalLinearSolveFailed, true
        return zero(scale)
    end
    return _box_vector(result.u) .* .!iszero.(scale)
end

function _box_lsq(cache, J, b, scale, diagonal, radius = Inf; lower_rhs = zero(scale))
    p = _box_linear_solve!(cache, J, b, scale, diagonal, lower_rhs)
    (cache.force_stop || LinearAlgebra.norm(p) <= radius) && return p
    g = scale .* _box_vector(adjoint(J) * b) + sqrt.(diagonal) .* lower_rhs
    q1 = -g / LinearAlgebra.norm(g)
    q2 = p - dot(q1, p) * q1
    Q = LinearAlgebra.norm(q2) > sqrt(eps(eltype(p))) * LinearAlgebra.norm(p) ?
        hcat(q1, q2 / LinearAlgebra.norm(q2)) : reshape(q1, :, 1)
    AQ = hcat((_box_vector(J * (scale .* q)) for q in eachcol(Q))...)
    H = transpose(AQ) * AQ + transpose(Q) * (diagonal .* Q)
    eig = LinearAlgebra.eigen(LinearAlgebra.Symmetric(H))
    values = max.(eig.values, zero(eltype(p)))
    rhs = transpose(eig.vectors) * (transpose(Q) * g)
    step(lambda) = -(eig.vectors * (rhs ./ (values .+ lambda)))
    lo = zero(eltype(p))
    hi = max(one(lo), LinearAlgebra.norm(rhs) / radius)
    for _ in 1:60
        mid = lo / 2 + hi / 2
        if LinearAlgebra.norm(step(mid)) > radius
            lo = mid
        else
            hi = mid
        end
        hi - lo <= eps(eltype(p)) * max(one(hi), hi) && break
    end
    return Q * step(hi)
end
