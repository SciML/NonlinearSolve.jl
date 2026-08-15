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

"""
    compose_precondition(f, pre, ::Val{iip})

Compose the left preconditioner `G` into the residual of the problem function `f`, returning
the composed function.

This is the extension point for function types that wrap a `NonlinearFunction`. Composition
cannot be done generically by rewriting the innermost callable: for a wrapper, the field
holding the callable is a whole `NonlinearFunction` carrying its own `jac`, `jac_prototype`
and `resid_prototype`, and replacing it with a bare composed callable silently discards them.
Types that know how to compose safely define a method; everything else is refused.
"""
function compose_precondition end

"""
    has_composed_precondition(f)

Whether `f`'s residual has already been composed with a left preconditioner. This is what
keeps [`transform_conditioned_problem`](@ref) idempotent across the nested
`solve`/`init`/`__solve` entry points, so define it alongside any
[`compose_precondition`](@ref) method that composes into a different place than the default.
"""
has_composed_precondition(f) = hasfield(typeof(f), :f) && f.f isa PreconditionWrapper

function compose_precondition(
        f::SciMLBase.NonlinearFunction, pre, ::Val{iip}
    ) where {iip}
    # Composing `G` changes the residual but leaves `f.jac`/`f.jvp`/`f.vjp` describing the
    # *uncomposed* map, so the solver would differentiate one function while evaluating
    # another — `J·δu = -G(f)` is not a Newton step for `G∘f`. Earlier versions did exactly
    # that and returned a converged-looking wrong answer, so this refuses instead. Compose
    # `G` into your residual and its derivative yourself if you need both.
    if SciMLBase.has_jac(f) || SciMLBase.has_jvp(f) || SciMLBase.has_vjp(f)
        throw(
            ArgumentError(
                "`precondition` cannot be combined with an analytic `jac`, `jvp` or `vjp`. \
                 The preconditioner is composed into the residual, but the supplied \
                 derivative still describes the uncomposed residual, so the two would \
                 disagree and the solve would converge to the wrong point. Compose the \
                 preconditioner into your residual and its derivative yourself, or drop the \
                 analytic derivative and let the solver differentiate the composed residual."
            )
        )
    end
    # Unwrap AutoSpecializeCallable before composing: the jacobian construction's Enzyme
    # unwrap path checks `is_fw_wrapped(prob.f.f)`, which cannot see a FunctionWrapper
    # hidden inside the composition.
    raw_f = is_fw_wrapped(f.f) ? get_raw_f(f.f) : f.f
    return @set f.f = PreconditionWrapper{iip}(raw_f, pre)
end

function compose_precondition(f::SciMLBase.AbstractNonlinearFunction, pre, ::Val)
    throw(
        ArgumentError(
            "`precondition` is not supported for a `$(nameof(typeof(f)))`. Composing the \
             preconditioner would have to rewrite the wrapped residual, which discards the \
             `jac`/`jac_prototype` the wrapper carries. Define \
             `NonlinearSolveBase.compose_precondition` for this function type, or compose \
             the preconditioner into your residual yourself."
        )
    )
end

"""
    AppliedInitialCorrection(corrector)

Marker recording that the `postcondition` corrector has already been applied to a problem's
initial guess. It forwards every call to the corrector it wraps.

`transform_conditioned_problem` runs at more than one funnel — `solve_call`, so that
algorithms with their own `__solve` see the composed problem, and again in `__solve` and
`init_call` — and a polyalgorithm re-enters `__solve` once per subsolver. Without a marker,
`H(u0, u0, p, nothing)` is applied once per pass, which for a non-idempotent corrector moves
the starting point somewhere the user never asked for.
"""
struct AppliedInitialCorrection{P}
    corrector::P
end

(applied::AppliedInitialCorrection)(u, u_prev, p, cache) =
    applied.corrector(u, u_prev, p, cache)

function initial_correction_applied(prob)
    has_kwargs(prob) || return false
    return get(prob.kwargs, :postcondition, nothing) isa AppliedInitialCorrection
end

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
    PostconditionSpace

Enum declaring which coordinates a `postcondition` corrector is written in when the
problem also carries `lb`/`ub` bounds. Bounds are handled by reparameterizing the
iterate to an unconstrained variable, so the two coordinate systems differ.

  - `PostconditionSpace.Original` (the default, and what a bare corrector gets): `H`
    sees the original bounded variable. The iterate is mapped back through the bounds
    transform, corrected, and mapped forward again at every commit point, so a limiting
    rule written for a physical quantity — a junction voltage in volts, a saturation in
    `[0, 1]` — means what it says. A correction landing exactly on a bound is nudged into
    the interior before the inverse map, since the bound itself is at infinity in the
    transformed variable.
  - `PostconditionSpace.Transformed`: `H` sees the unconstrained variable the solver
    iterates on. Use this for corrections that are statements about the solver's step
    (damping a raw update, say), not about the model. The initial guess is then left
    uncorrected, since it is still in the original coordinates when the corrector would
    run.

Without `lb`/`ub` the two are identical.
"""
EnumX.@enumx PostconditionSpace begin
    Original
    Transformed
end

"""
    PostconditionSpecifier(corrector; space = PostconditionSpace.Original)

Wrapper for the `postcondition` solver option that declares which coordinates the iterate
corrector `H(u_proposed, u_prev, p, cache)` is written in when the problem also carries
`lb`/`ub` bounds. See [`PostconditionSpace`](@ref) for the meaning of each value.

Without `lb`/`ub` the two spaces are identical and the wrapper is unnecessary.

```julia
solve(prob, NewtonRaphson(); postcondition = H)   # PostconditionSpace.Original
solve(
    prob, NewtonRaphson();
    postcondition = PostconditionSpecifier(H; space = PostconditionSpace.Transformed)
)
```
"""
struct PostconditionSpecifier{space, F}
    corrector::F
end

function PostconditionSpecifier(
        corrector; space::PostconditionSpace.T = PostconditionSpace.Original
    )
    return PostconditionSpecifier{space, typeof(corrector)}(corrector)
end

function (spec::PostconditionSpecifier)(u, u_prev, p, cache)
    return spec.corrector(u, u_prev, p, cache)
end

# Bare correctors are Original: with no bounds the two spaces coincide, and with
# bounds the original variable is the one the corrector was written for.
postcondition_space(_) = PostconditionSpace.Original
postcondition_space(::PostconditionSpecifier{space}) where {space} = space
postcondition_space(applied::AppliedInitialCorrection) = postcondition_space(applied.corrector)

"""
    get_postcondition(prob, kwargs)

The iterate corrector `H(u_proposed, u_prev, p, cache)` in effect for this solve, or
`nothing`. It may be wrapped in a [`PostconditionSpecifier`](@ref).
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
    if get_precondition(prob, kwargs) !== nothing && !has_composed_precondition(prob.f)
        return true
    end
    # Deliberately not gated on `initial_correction_applied`: the transform also reports an
    # unsupported `postcondition`, and the first funnel to run may not know the algorithm yet
    # (`solve(prob; postcondition = H)` resolves it later). Re-entering is cheap — the
    # transform skips the correction itself via the same marker.
    return get_postcondition(prob, kwargs) !== nothing
end

"""
    transform_conditioned_problem(prob, alg, kwargs)

Compose the `precondition` option into the problem's residual and apply the
`postcondition` option once to the initial guess as `H(u0, u0, p, nothing)`, so solves
start from a corrected iterate. Reports a `postcondition` combined with an algorithm that
does not apply it (see [`supports_postcondition`](@ref)) through the
`unsupported_postcondition` verbosity toggle, which raises by default.

The initial guess is skipped only for a `PostconditionSpace.Transformed` corrector on a
bounded problem: `prob.u0` is still in the original coordinates here, since this pass
runs before the bounds transform.
"""
function transform_conditioned_problem(prob, alg, kwargs)
    pre = get_precondition(prob, kwargs)
    post = get_postcondition(prob, kwargs)

    if post !== nothing &&
            alg !== nothing && alg isa AbstractNonlinearSolveAlgorithm &&
            !supports_postcondition(alg)
        verbose = get(kwargs, :verbose, NonlinearVerbosity())
        @SciMLMessage(
            "The `postcondition` solver option is not supported by \
            $(typeof(alg).name.name), so the corrector will not be applied. Use a \
            solver that applies iterate corrections (e.g. the native \
            NonlinearSolve.jl first-order, quasi-Newton, or spectral methods).",
            verbose, :unsupported_postcondition
        )
    end

    # Skip the initial-guess correction for a Transformed-space corrector on a bounded
    # problem: the bounds transform runs after this pass, so `prob.u0` is still in the
    # original coordinates and applying H here would mix spaces.
    skip_initial = post === nothing || initial_correction_applied(prob) || (
        postcondition_space(post) === PostconditionSpace.Transformed &&
            hasfield(typeof(prob), :lb) && hasfield(typeof(prob), :ub) &&
            (prob.lb !== nothing || prob.ub !== nothing) &&
            needs_bounds_transform(prob, alg)
    )

    u0 = if skip_initial
        prob.u0
    elseif SciMLBase.isinplace(prob)
        u0c = copy(prob.u0)
        _apply_postcondition!!(post, u0c, prob.u0, prob.p, nothing, Val(true))
    else
        _apply_postcondition!!(post, prob.u0, prob.u0, prob.p, nothing, Val(false))
    end

    # Record the initial-guess correction on the problem so the later funnels — and a
    # polyalgorithm's per-subsolver `__solve` — do not apply it again.
    newkwargs = skip_initial ? missing : _mark_initial_correction(prob, post)

    if pre === nothing || has_composed_precondition(prob.f)
        return (u0 === prob.u0 && newkwargs === missing) ? prob :
            remake(prob; u0, kwargs = newkwargs)
    end

    composed = compose_precondition(prob.f, pre, Val(SciMLBase.isinplace(prob)))
    return remake(prob; f = composed, u0, kwargs = newkwargs)
end

function _mark_initial_correction(prob, post)
    base = has_kwargs(prob) ? values(prob.kwargs) : NamedTuple()
    return merge(base, (; postcondition = AppliedInitialCorrection(post)))
end

"""
    apply_postcondition!!(u, u_prev, cache)

Apply the solve's `postcondition` corrector to the just-committed iterate `u`, given the
previous accepted iterate `u_prev`. Returns the corrected iterate (`u` itself for
in-place problems). Solver families must call this at every iterate-commit point *before*
evaluating the residual or testing convergence there, so residuals and Jacobians stay
consistent with the corrected iterates.

Correctors always take the solver cache as their fourth argument; a corrector that does
not need it simply ignores the parameter. Only the documented cache accessors (`get_u`,
`get_fu`, `get_nsteps`, `get_abstol`, `get_reltol`) should be used on it. The argument is
`nothing` for the initial-guess correction, which runs before any cache exists.

On a bounded problem the solver iterates on the unconstrained variable produced by the
`lb`/`ub` transform, so `u` here is in transformed coordinates. Unless the corrector was
declared `PostconditionSpace.Transformed` (see [`PostconditionSpecifier`](@ref)), it is
applied in the original bounded variable: the iterates are mapped back, corrected, and
mapped forward again.
"""
function apply_postcondition!!(u, u_prev, cache)
    post = get_postcondition(cache)
    post === nothing && return u
    iip = Val(SciMLBase.isinplace(cache.prob))
    bw = bounded_wrapper(cache)
    (bw === nothing || postcondition_space(post) === PostconditionSpace.Transformed) &&
        return _apply_postcondition!!(post, u, u_prev, cache.prob.p, cache, iip)
    return _apply_postcondition_bounded!!(post, u, u_prev, cache.prob.p, cache, iip, bw)
end

# In-place and out-of-place are dispatched rather than branched on so the unused
# convention's return type does not leak into the caller's inference.
function _apply_postcondition!!(post::F, u, u_prev, p, cache, ::Val{true}) where {F}
    post(u, u_prev, p, cache)
    return u
end

function _apply_postcondition!!(post::F, u, u_prev, p, cache, ::Val{false}) where {F}
    return post(u, u_prev, p, cache)
end

# Apply the corrector in the original bounded variable while the solver iterates on the
# unconstrained one. `_clamp_to_bounds` is load-bearing on the way back: a corrector that
# clamps exactly *onto* a bound sits at ±Inf in the transformed variable, so it has to be
# nudged into the interior first.
#
# The IIP path writes into the preallocated `BoundedWrapper` temps so the commit-point
# corrector is allocation-free. The OOP path still allocates the mapped inputs and the
# mapped return — that is the out-of-place contract.
function _apply_postcondition_bounded!!(
        post::F, u, u_prev, p, cache, ::Val{true}, bw
    ) where {F}
    u_orig = _bounds_tmp(bw.u_cache, u)
    u_prev_orig = _bounds_tmp(bw.u_prev_cache, u_prev)
    @. u_orig = _from_unbounded(u, bw.lb, bw.ub)
    @. u_prev_orig = _from_unbounded(u_prev, bw.lb, bw.ub)
    post(u_orig, u_prev_orig, p, cache)
    @. u = _to_unbounded(_clamp_to_bounds(u_orig, bw.lb, bw.ub), bw.lb, bw.ub)
    return u
end

function _apply_postcondition_bounded!!(
        post::F, u, u_prev, p, cache, ::Val{false}, bw
    ) where {F}
    u_orig = _from_unbounded.(u, bw.lb, bw.ub)
    u_prev_orig = _from_unbounded.(u_prev, bw.lb, bw.ub)
    u_new = post(u_orig, u_prev_orig, p, cache)
    return _to_unbounded.(_clamp_to_bounds.(u_new, bw.lb, bw.ub), bw.lb, bw.ub)
end
