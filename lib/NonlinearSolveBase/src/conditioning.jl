# Nonlinear preconditioning hooks (`NonlinearFunction` `precondition`/`postcondition`).
#
# `precondition` (a left preconditioner `G`) is handled as a problem transformation: the
# residual is replaced by the root-equivalent composition `u -> G(f(u, p), u, p)` before
# the solver cache is built, so every consumer (function evaluations, AD Jacobians,
# line-search merit functions, termination criteria) sees the composed map. `postcondition`
# (a right preconditioner / iterate corrector `H`) needs solver-loop support and is applied
# by the solver families at every iterate-commit point via `apply_postcondition!!`.

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

# Marker left in the `precondition` field once the hook has been composed into `f`.
# It cannot be cleared to `nothing` instead: `remake(prob; f = new_f)` merges function
# fields and lets `nothing` fields of `new_f` fall back to the old function's values,
# which would resurrect the hook and re-wrap on the next transform pass.
struct ComposedPrecondition end

"""
    has_precondition(prob)

Whether the problem function carries a not-yet-composed `precondition` hook.
"""
function has_precondition(prob)
    f = prob.f
    return hasfield(typeof(f), :precondition) && f.precondition !== nothing &&
        !(f.precondition isa ComposedPrecondition)
end

"""
    has_postcondition(prob)

Whether the problem function carries a `postcondition` iterate-correction hook.
"""
function has_postcondition(prob)
    f = prob.f
    return hasfield(typeof(f), :postcondition) && f.postcondition !== nothing
end

"""
    supports_postcondition(alg)

Trait declaring whether a solver algorithm applies `NonlinearFunction.postcondition` at
its iterate-commit points. Algorithms without support must not silently ignore the hook,
so `transform_conditioned_problem` throws for them.
"""
supports_postcondition(alg) = false

"""
    needs_conditioning(prob)

Whether `transform_conditioned_problem` must run on this problem before solving, i.e.
whether an applicable problem type carries an active nonlinear preconditioning hook.
"""
function needs_conditioning(prob)
    return (
        prob isa SciMLBase.NonlinearProblem ||
            prob isa SciMLBase.NonlinearLeastSquaresProblem ||
            prob isa SciMLBase.ImmutableNonlinearProblem
    ) && (has_precondition(prob) || has_postcondition(prob))
end

"""
    transform_conditioned_problem(prob, alg)

Compose the `precondition` hook into the problem function (marking the hook consumed so
the transform is idempotent) and apply the `postcondition` hook once to the initial
guess as `H(u0, u0, p)` so solves start from a corrected/feasible iterate. Throws an
`ArgumentError` when a `postcondition` is combined with an algorithm that does not
apply it (see `supports_postcondition`) or with `lb`/`ub` bounds.
"""
function transform_conditioned_problem(prob, alg)
    if has_postcondition(prob)
        if alg !== nothing && alg isa AbstractNonlinearSolveAlgorithm &&
                !supports_postcondition(alg)
            throw(ArgumentError("`NonlinearFunction.postcondition` is not supported by \
                $(typeof(alg).name.name). Use a solver that applies iterate corrections \
                (e.g. the native NonlinearSolve.jl first-order, quasi-Newton, or spectral \
                methods)."))
        end
        if hasfield(typeof(prob), :lb) && hasfield(typeof(prob), :ub) &&
                (prob.lb !== nothing || prob.ub !== nothing)
            throw(ArgumentError("`NonlinearFunction.postcondition` cannot be combined \
                with `lb`/`ub` bounds: the bounds transform changes the iterate \
                coordinates the hook would act on. Enforce the bounds inside the \
                `postcondition` instead."))
        end
    end

    u0 = if has_postcondition(prob)
        post = prob.f.postcondition
        if SciMLBase.isinplace(prob)
            u0c = copy(prob.u0)
            post(u0c, prob.u0, prob.p)
            u0c
        else
            post(prob.u0, prob.u0, prob.p)
        end
    else
        prob.u0
    end

    has_precondition(prob) || return remake(prob; u0)

    orig_f = prob.f
    # Unwrap AutoSpecializeCallable before composing: the jacobian construction's
    # Enzyme unwrap path checks `is_fw_wrapped(prob.f.f)`, which cannot see a
    # FunctionWrapper hidden inside the composition.
    raw_f = is_fw_wrapped(orig_f.f) ? get_raw_f(orig_f.f) : orig_f.f
    wrapped = PreconditionWrapper{SciMLBase.isinplace(prob)}(raw_f, orig_f.precondition)

    new_f = @set orig_f.f = wrapped
    @set! new_f.precondition = ComposedPrecondition()

    return remake(prob; f = new_f, u0)
end

"""
    apply_postcondition!!(u, u_prev, prob)

Apply `prob.f.postcondition` to the just-committed iterate `u` given the previous
accepted iterate `u_prev`, following the problem's in-place convention. Returns the
corrected iterate (`u` itself for in-place problems). Solver families must call this at
every iterate-commit point *before* evaluating the residual or testing convergence there,
so residuals and Jacobians stay consistent with the corrected iterates.
"""
function apply_postcondition!!(u, u_prev, prob)
    has_postcondition(prob) || return u
    post = prob.f.postcondition
    if SciMLBase.isinplace(prob)
        post(u, u_prev, prob.p)
        return u
    else
        return post(u, u_prev, prob.p)
    end
end
