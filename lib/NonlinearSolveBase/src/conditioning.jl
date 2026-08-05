# Nonlinear preconditioning: the `precondition` (left preconditioner `G`) and
# `postcondition` (right preconditioner / iterate corrector `H`) solver options.
#
# Both are ordinary solve keywords rather than properties of the `NonlinearFunction`, so
# they can be supplied late — at `solve`/`init`, or carried on the problem's `kwargs` and
# merged in like any other option (solve-time wins, matching `alias`). `G` is applied as a
# problem transformation before the cache is built, so every consumer (residual
# evaluations, AD Jacobians, line-search merit functions, termination criteria) sees the
# composed map; `H` is read from the cache at each iterate-commit point.

@concrete struct PreconditionWrapper{iip}
    f
    precondition
end

function (w::PreconditionWrapper{false})(u, p)
    return w.precondition(w.f(u, p), u, p)
end

function (w::PreconditionWrapper{true})(resid, u, p)
    w.f(resid, u, p)
    w.precondition(resid, u, p)
    return resid
end

SciMLBase.isinplace(w::PreconditionWrapper{iip}) where {iip} = iip

# Solve keywords take precedence over the problem's stored keywords, matching `alias`.
function _conditioning_option(prob, kwargs, key::Symbol)
    val = get(kwargs, key, nothing)
    val === nothing || return val
    has_kwargs(prob) || return nothing
    return get(prob.kwargs, key, nothing)
end

"""
    get_precondition(prob, kwargs)

The left preconditioner `G(fu, u, p)` in effect for this solve, or `nothing`.
"""
get_precondition(prob, kwargs) = _conditioning_option(prob, kwargs, :precondition)

"""
    get_postcondition(prob, kwargs)

The iterate corrector `H(u_proposed, u_prev, p[, cache])` in effect for this solve, or
`nothing`.
"""
get_postcondition(prob, kwargs) = _conditioning_option(prob, kwargs, :postcondition)

"""
    get_postcondition(cache)

The iterate corrector in effect for an initialized solver cache, read from the keywords
the cache was built with.
"""
get_postcondition(cache) = get(cache.kwargs, :postcondition, nothing)

"""
    supports_postcondition(alg)

Trait declaring whether a solver algorithm applies the `postcondition` corrector at its
iterate-commit points. Algorithms without support must not silently ignore the option, so
`transform_conditioned_problem` throws for them.
"""
supports_postcondition(alg) = false

"""
    needs_conditioning(prob, kwargs)

Whether `transform_conditioned_problem` must run before solving. A problem whose residual
has already been composed with its preconditioner is skipped, which keeps the transform
idempotent across the nested `solve`/`init`/`__solve` entry points.
"""
function needs_conditioning(prob, kwargs)
    (
        prob isa SciMLBase.NonlinearProblem ||
            prob isa SciMLBase.NonlinearLeastSquaresProblem ||
            prob isa SciMLBase.ImmutableNonlinearProblem
    ) || return false
    if get_precondition(prob, kwargs) !== nothing &&
            !(hasfield(typeof(prob.f), :f) && prob.f.f isa PreconditionWrapper)
        return true
    end
    return get_postcondition(prob, kwargs) !== nothing
end

"""
    transform_conditioned_problem(prob, alg, kwargs)

Compose the `precondition` option into the problem's residual and apply the
`postcondition` option once to the initial guess as `H(u0, u0, p)`, so solves start from a
corrected iterate. Throws an `ArgumentError` when a `postcondition` is combined with an
algorithm that does not apply it (see [`supports_postcondition`](@ref)) or with `lb`/`ub`
bounds.
"""
function transform_conditioned_problem(prob, alg, kwargs)
    pre = get_precondition(prob, kwargs)
    post = get_postcondition(prob, kwargs)

    if post !== nothing
        if alg !== nothing && alg isa AbstractNonlinearSolveAlgorithm &&
                !supports_postcondition(alg)
            throw(
                ArgumentError(
                    "the `postcondition` solver option is not supported by \
                    $(typeof(alg).name.name). Use a solver that applies iterate \
                    corrections (e.g. the native NonlinearSolve.jl first-order, \
                    quasi-Newton, or spectral methods)."
                )
            )
        end
        if hasfield(typeof(prob), :lb) && hasfield(typeof(prob), :ub) &&
                (prob.lb !== nothing || prob.ub !== nothing)
            throw(
                ArgumentError(
                    "the `postcondition` solver option cannot be combined with `lb`/`ub` \
                    bounds: the bounds transform changes the iterate coordinates the \
                    corrector would act on. Enforce the bounds inside the `postcondition` \
                    instead."
                )
            )
        end
    end

    u0 = if post === nothing
        prob.u0
    elseif SciMLBase.isinplace(prob)
        u0c = copy(prob.u0)
        _apply_postcondition!!(post, u0c, prob.u0, prob.p, nothing, true)
    else
        _apply_postcondition!!(post, prob.u0, prob.u0, prob.p, nothing, false)
    end

    if pre === nothing || (hasfield(typeof(prob.f), :f) && prob.f.f isa PreconditionWrapper)
        return u0 === prob.u0 ? prob : remake(prob; u0)
    end

    orig_f = prob.f
    # Unwrap AutoSpecializeCallable before composing: the jacobian construction's Enzyme
    # unwrap path checks `is_fw_wrapped(prob.f.f)`, which cannot see a FunctionWrapper
    # hidden inside the composition.
    raw_f = is_fw_wrapped(orig_f.f) ? get_raw_f(orig_f.f) : orig_f.f
    wrapped = PreconditionWrapper{SciMLBase.isinplace(prob)}(raw_f, pre)

    return remake(prob; f = @set(orig_f.f = wrapped), u0)
end

"""
    apply_postcondition!!(u, u_prev, cache)

Apply the solve's `postcondition` corrector to the just-committed iterate `u`, given the
previous accepted iterate `u_prev`. Returns the corrected iterate (`u` itself for
in-place problems). Solver families must call this at every iterate-commit point *before*
evaluating the residual or testing convergence there, so residuals and Jacobians stay
consistent with the corrected iterates.

Correctors may opt into solver-state access by accepting a fourth argument,
`H(u_proposed, u_prev, p, cache)`; when both arities exist the four-argument form is
preferred. Only the documented cache accessors (`get_u`, `get_fu`, `get_nsteps`,
`get_abstol`, `get_reltol`) should be used on it. The argument is `nothing` for the
initial-guess correction, which runs before any cache exists.
"""
function apply_postcondition!!(u, u_prev, cache)
    post = get_postcondition(cache)
    post === nothing && return u
    return _apply_postcondition!!(
        post, u, u_prev, cache.prob.p, cache, SciMLBase.isinplace(cache.prob)
    )
end

function _apply_postcondition!!(post::F, u, u_prev, p, cache, iip::Bool) where {F}
    if applicable(post, u, u_prev, p, cache)
        iip && (post(u, u_prev, p, cache); return u)
        return post(u, u_prev, p, cache)
    end
    iip && (post(u, u_prev, p); return u)
    return post(u, u_prev, p)
end
