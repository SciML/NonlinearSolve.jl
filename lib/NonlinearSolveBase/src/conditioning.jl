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
    if get_precondition(prob, kwargs) !== nothing &&
            !(hasfield(typeof(prob.f), :f) && prob.f.f isa PreconditionWrapper)
        return true
    end
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

    verbose = get(kwargs, :verbose, NonlinearVerbosity())
    corrects_transformed = false
    if post !== nothing
        if alg !== nothing && alg isa AbstractNonlinearSolveAlgorithm &&
                !supports_postcondition(alg)
            @SciMLMessage(
                "The `postcondition` solver option is not supported by \
                $(typeof(alg).name.name), so the corrector will not be applied. Use a \
                solver that applies iterate corrections (e.g. the native \
                NonlinearSolve.jl first-order, quasi-Newton, or spectral methods).",
                verbose, :unsupported_postcondition
            )
        end
        if postcondition_space(post) === PostconditionSpace.Transformed &&
                hasfield(typeof(prob), :lb) && hasfield(typeof(prob), :ub) &&
                (prob.lb !== nothing || prob.ub !== nothing) &&
                needs_bounds_transform(prob, alg)
            # The corrector was declared in the coordinates the bounds transform iterates
            # on, and that transform runs after this pass — `prob.u0` here is still the
            # original bounded variable, so the initial-guess correction is skipped to
            # keep every application in the same space.
            corrects_transformed = true
            @SciMLMessage(
                "The `postcondition` corrector was declared with \
                `space = PostconditionSpace.Transformed`, so it is imposed on the \
                unconstrained variable the `lb`/`ub` transform iterates on rather than on \
                the original bounded variable, and the initial-guess correction is \
                skipped (the initial guess is still in the original coordinates at that \
                point).",
                verbose, :postcondition_bounds_transform
            )
        end
    end

    u0 = if post === nothing || corrects_transformed
        prob.u0
    elseif SciMLBase.isinplace(prob)
        u0c = copy(prob.u0)
        _apply_postcondition!!(post, u0c, prob.u0, prob.p, nothing, Val(true))
    else
        _apply_postcondition!!(post, prob.u0, prob.u0, prob.p, nothing, Val(false))
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
    return _apply_postcondition_bounded!!(
        post, u, u_prev, cache.prob.p, cache, iip, bw.lb, bw.ub
    )
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
function _apply_postcondition_bounded!!(
        post::F, u, u_prev, p, cache, ::Val{true}, lb, ub
    ) where {F}
    u_orig = _from_unbounded.(u, lb, ub)
    u_prev_orig = _from_unbounded.(u_prev, lb, ub)
    post(u_orig, u_prev_orig, p, cache)
    @. u = _to_unbounded(_clamp_to_bounds(u_orig, lb, ub), lb, ub)
    return u
end

function _apply_postcondition_bounded!!(
        post::F, u, u_prev, p, cache, ::Val{false}, lb, ub
    ) where {F}
    u_orig = _from_unbounded.(u, lb, ub)
    u_prev_orig = _from_unbounded.(u_prev, lb, ub)
    u_new = post(u_orig, u_prev_orig, p, cache)
    return _to_unbounded.(_clamp_to_bounds.(u_new, lb, ub), lb, ub)
end
