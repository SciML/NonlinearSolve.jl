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
    eps_frac = eps(typeof(u))^(3 / 4)
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

# Wrapper that contains the bounds and caches for mapped-back iterates.
# `u_cache` is the residual-evaluation temporary (and the postcondition `u` buffer);
# `u_prev_cache` is a second temporary so the original-space postcondition path can map
# both the proposed and previous iterates without allocating.
@concrete struct BoundedWrapper{isinplace}
    f
    lb
    ub
    original_u0
    u_cache
    u_prev_cache
end

@inline function _bounds_tmp(cache, u)
    return cache isa FixedSizeDiffCache ? get_tmp(cache, u) : cache
end

function _transform_u(w::BoundedWrapper, u)
    tmp = _bounds_tmp(w.u_cache, u)
    @. tmp = _from_unbounded(u, w.lb, w.ub)
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

# The `BoundedWrapper` a cache's problem function was wrapped in, or `nothing` when the
# solve is not running in transformed coordinates. Every check is on types, so the whole
# lookup constant-folds away for problems without bounds.
@inline function bounded_wrapper_from_problem(prob)
    wrapped = hasfield(typeof(prob), :f) && hasfield(typeof(prob.f), :f) &&
        prob.f.f isa BoundedWrapper
    return wrapped ? prob.f.f : nothing
end

@inline function bounded_wrapper(cache)
    return bounded_wrapper_from_problem(cache.prob)
end

# Check if bounds transform is needed for a given problem and algorithm.
function needs_bounds_transform(_prob, alg)
    return (
        _prob isa SciMLBase.NonlinearProblem ||
            _prob isa SciMLBase.NonlinearLeastSquaresProblem
    ) &&
        (hasfield(typeof(_prob), :lb) && hasfield(typeof(_prob), :ub)) &&
        (_prob.lb !== nothing || _prob.ub !== nothing) &&
        (isnothing(alg) || !SciMLBase.allowsbounds(alg))
end

# Wrap a problem function with bounds into a BoundedWrapper with no bounds. In a
# nutshell, we transform a parameter `p` with bounds `lb` and `ub` into an
# unbounded parameter `t` using the logistic function to map all values of `t`
# into the interval (lb, ub).
function transform_bounded_problem(prob, alg)
    # `initialization_data` (e.g. from ModelingToolkit) is defined in the original,
    # bounded coordinates. It must run *before* the transform so the resulting `u0`/`p`
    # are consistent; running it on the transformed problem would interpret the unbounded
    # iterate `t` as a bounded state and corrupt the solution. We run it here and strip it
    # from the transformed function below so it isn't re-run in `t`-space.
    prob = run_bounded_initialization(prob, alg)

    lb, ub = _normalize_bounds(prob.lb, prob.ub, prob.u0)

    # Clamp u0 into the interior of the bounds so that _to_unbounded doesn't hit log(0)
    # or log(negative). We nudge by a small fraction of the interval width.
    u0_clamped = _clamp_to_bounds.(prob.u0, lb, ub)
    u0_transformed = _to_unbounded.(u0_clamped, lb, ub)

    # PreallocationTools is only supported by ForwardDiff so we only use
    # FixedSizeDiffCache if we're using ForwardDiff. Not every algorithm has an
    # `autodiff` field (e.g. `QuasiNewtonAlgorithm`), so guard the access.
    alg_ad = alg !== nothing && hasproperty(alg, :autodiff) ? alg.autodiff : nothing
    make_u_cache = if alg_ad === nothing || alg_ad isa AutoForwardDiff
        () -> FixedSizeDiffCache(prob.u0)
    else
        () -> similar(prob.u0)
    end
    u_cache = make_u_cache()
    u_prev_cache = make_u_cache()

    orig_f = prob.f
    # Unwrap AutoSpecializeCallable before wrapping in BoundedWrapper.
    # BoundedWrapper transforms arguments, so the FunctionWrapper signatures won't match.
    unwrapped_orig_f = if is_fw_wrapped(orig_f.f)
        @set orig_f.f = get_raw_f(orig_f.f)
    else
        orig_f
    end
    wrapped = BoundedWrapper{SciMLBase.isinplace(prob)}(
        unwrapped_orig_f, lb, ub, copy(prob.u0), u_cache, u_prev_cache
    )

    new_f = if orig_f isa NonlinearFunction
        @set orig_f.f = wrapped
    else
        wrapped
    end

    transformed_prob = remake(
        prob; f = new_f, u0 = u0_transformed, lb = nothing, ub = nothing,
        build_initializeprob = Val{false}
    )

    return transformed_prob
end

function bounded_retry_u0(bw, original_u0)
    retry_u0 = similar(original_u0)
    @inbounds for i in eachindex(retry_u0)
        lb, ub, u = bw.lb[i], bw.ub[i], original_u0[i]
        retry_u0[i] = if isfinite(lb) && isfinite(ub)
            lb + (ub - lb) / 2
        elseif isfinite(lb)
            lb + 5 * max(abs(u - lb), one(u))
        elseif isfinite(ub)
            ub - 5 * max(abs(ub - u), one(u))
        else
            u
        end
    end
    return _to_unbounded.(retry_u0, bw.lb, bw.ub)
end

function bounded_retry_converged(prob, sol)
    SciMLBase.successful_retcode(sol) && return true
    sol.retcode == ReturnCode.Stalled || return false
    threshold = 2 * (get_abstol(prob) + get_reltol(prob) * max(one(eltype(prob.u0)), Linf_NORM(sol.u)))
    return Linf_NORM(sol.resid) ≤ threshold
end

function bounded_retry_solution(prob, sol, alg, args, kwargs)
    if bounded_retry_converged(prob, sol)
        sol.retcode == ReturnCode.Stalled && (@set! sol.retcode = ReturnCode.Success)
        return sol
    end
    hasfield(typeof(prob), :f) && hasfield(typeof(prob.f), :f) || return sol
    bw = prob.f.f isa BoundedWrapper ? prob.f.f : nothing
    bw === nothing && return sol

    retry_u0 = bounded_retry_u0(bw, bw.original_u0)
    retry_prob = remake(prob; u0 = retry_u0)
    retry_cache = SciMLBase.__init(retry_prob, alg, args...; kwargs...)
    retry_sol = CommonSolve.solve!(retry_cache)
    if bounded_retry_converged(retry_prob, retry_sol)
        retry_sol.retcode == ReturnCode.Stalled && (@set! retry_sol.retcode = ReturnCode.Success)
        @set! retry_sol.prob = remake(retry_sol.prob; u0 = bw.original_u0)
        return retry_sol
    end
    return sol
end

# Run problem initialization (if any) in the original bounded coordinates so that the
# resulting `u0`/`p` are consistent before the bounds transform is applied. Returns the
# problem unchanged when there is no `OverrideInitData` to run.
function run_bounded_initialization(prob, alg)
    SciMLBase.has_initialization_data(prob.f) || return prob
    prob.f.initialization_data isa SciMLBase.OverrideInitData || return prob
    iip = SciMLBase.isinplace(prob)
    u0, p, _ = SciMLBase.get_initial_values(
        prob, prob, prob.f, SciMLBase.OverrideInit(), Val(iip);
        nlsolve_alg = alg, abstol = get_abstol(prob), reltol = get_reltol(prob)
    )
    return remake(prob; u0, p, build_initializeprob = Val{false})
end
