"""
    ArcLengthContinuation(; inner = nothing, initial_step_factor = 0.1,
        adaptive = true, min_ds = nothing, max_step_factor = 1.0,
        expand_factor = 2.0, expand_threshold = 2, max_angle = π / 6,
        predictor = :secant, autodiff = nothing, maxsteps = 10000)

Pseudo-arclength continuation solver for a `SciMLBase.HomotopyProblem`. Unlike
[`HomotopySweep`](@ref), which marches the scalar parameter ``λ`` monotonically, this
solver tracks the solution curve ``H(u, λ) = 0`` parameterized by arclength ``s`` in the
augmented ``(u, λ)`` space. Each step takes a predictor step along the path and corrects
it by solving the *augmented* ``(n+1)``-dimensional system

```
H(u, λ)                 = 0          # n equations
τ ⋅ ([u; λ] - x₀) - Δs  = 0          # Keller pseudo-arclength constraint
```

with the `inner` solver. Because ``λ`` is a free variable of the corrector (not held
fixed), the augmented Jacobian stays nonsingular at *turning points* (folds) where
``∂H/∂u`` is singular and ``λ`` is non-monotone along the path. This lets the solver
round folds that defeat natural-parameter continuation — the canonical reason a sweep
"fails to reach `λ = 1`" when a real solution at `λ = 1` does exist but only on a branch
reachable by going around a fold.

The target is the point on the curve where ``λ = λspan[2]``: the solver follows the path
until a step brackets that ``λ``, then performs one final ``λ``-fixed correction to land
on it exactly.

Keyword arguments:

  - `inner`: the inner nonlinear algorithm used for both the initial on-curve correction
    and the augmented corrector; `nothing` selects NonlinearSolve's default polyalgorithm.
  - `initial_step_factor`: the initial arclength step `Δs` as a fraction of the `λspan`
    width. Must be in `(0, 1]`.
  - `adaptive`: when `true` (default), a corrector failure halves `Δs` and retries from
    the last accepted point (down to a floor of `min_ds`), and `expand_threshold`
    consecutive successes grow `Δs` by `expand_factor` up to `max_step_factor` of the
    span.
  - `min_ds`: the smallest arclength step bisection may reach; `nothing` (default)
    resolves to `sqrt(eps(typeof(λ)))`.
  - `max_step_factor`: the largest arclength step, as a fraction of the `λspan` width.
    Must be in `(0, 1]`.
  - `expand_factor`: the `Δs` growth multiplier after `expand_threshold` consecutive
    successful steps. Must be ≥ 1; `1` disables expansion.
  - `expand_threshold`: consecutive successful steps required before `Δs` is expanded.
    Must be ≥ 1.
  - `max_angle`: the curvature control (radians, in `(0, π]`). A step is *rejected* and
    `Δs` halved when the path direction turns by more than `max_angle` between the
    previous and current accepted segments; `Δs` is only allowed to grow when the turn is
    below `max_angle / 3`. Because the solution curve is smooth in arclength even at a
    fold (the tangent rotates continuously), bounding the per-step turn forces small
    steps *through* a turning point while permitting large steps on straight stretches —
    and it is what prevents the secant predictor from overshooting onto a different branch
    ("path jumping"). This is the analogue of OpenModelica's homotopy bend parameter.
  - `predictor`: how the initial guess for each corrector is extrapolated along the path.
    `:secant` (default) uses the secant through the last two accepted points, bootstrapped
    from a pure-``λ`` step — derivative-free, but the bootstrap step cannot round a fold
    located within the very first step, and it is not curvature-checked. `:tangent`
    instead computes the true path tangent at the current point as the (oriented) null
    vector of the augmented Jacobian ``[∂H/∂u | ∂H/∂λ]``, which stays well-defined at a
    fold (where the tangent is vertical in ``λ``). The tangent is a higher-order predictor
    and is accurate from the first step, so it curvature-checks every step and can round a
    fold at the very start; the cost is one Jacobian factorization per step (see
    `autodiff`). This is the Euler tangent predictor of the classic path trackers and of
    OpenModelica's global homotopy.
  - `autodiff`: the automatic-differentiation backend (an `ADTypes.AbstractADType`) used
    to form the augmented Jacobian for the `:tangent` predictor; `nothing` (default)
    selects `AutoForwardDiff()`. Unused by the `:secant` predictor.
  - `maxsteps`: a hard cap on the total number of predictor-corrector attempts (including
    bisection retries). Required because the path is *not* monotone in `λ`, so a sweep
    that never reaches the target — a closed loop, or a branch escaping to infinity —
    would otherwise not terminate. Exceeding it returns a `ReturnCode.MaxIters` failure.

When the solver cannot reach `λspan[2]`, the returned solution carries a failure retcode
and its `u` is the last converged curve point.

With the default `:secant` predictor the continuation is derivative-free in the predictor;
the augmented corrector obtains the derivatives it needs through the inner solver's own
differentiation, exactly as a standard `NonlinearProblem` would. The `:tangent` predictor
additionally differentiates the homotopy through `autodiff` to build the augmented
Jacobian.
"""
@concrete struct ArcLengthContinuation <: AbstractNonlinearSolveAlgorithm
    inner
    initial_step_factor
    adaptive::Bool
    min_ds
    max_step_factor
    expand_factor
    expand_threshold::Int
    max_angle
    predictor::Symbol
    autodiff
    maxsteps::Int
end

function ArcLengthContinuation(;
        inner = nothing, initial_step_factor = 0.1, adaptive = true,
        min_ds = nothing, max_step_factor = 1.0, expand_factor = 2.0,
        expand_threshold = 2, max_angle = π / 6, predictor = :secant,
        autodiff = nothing, maxsteps = 10000
    )
    if !(0 < initial_step_factor <= 1)
        throw(
            ArgumentError(
                "ArcLengthContinuation `initial_step_factor` must be in (0, 1], got $initial_step_factor"
            )
        )
    end
    if min_ds !== nothing && min_ds <= 0
        throw(ArgumentError("ArcLengthContinuation `min_ds` must be positive, got $min_ds"))
    end
    if !(0 < max_step_factor <= 1)
        throw(
            ArgumentError(
                "ArcLengthContinuation `max_step_factor` must be in (0, 1], got $max_step_factor"
            )
        )
    end
    if expand_factor < 1
        throw(
            ArgumentError(
                "ArcLengthContinuation `expand_factor` must be ≥ 1 (1 disables expansion), got $expand_factor"
            )
        )
    end
    if expand_threshold < 1
        throw(
            ArgumentError(
                "ArcLengthContinuation `expand_threshold` must be ≥ 1, got $expand_threshold"
            )
        )
    end
    if !(0 < max_angle <= π)
        throw(
            ArgumentError(
                "ArcLengthContinuation `max_angle` must be in (0, π], got $max_angle"
            )
        )
    end
    if predictor !== :secant && predictor !== :tangent
        throw(
            ArgumentError(
                "ArcLengthContinuation `predictor` must be :secant or :tangent, got :$predictor"
            )
        )
    end
    if maxsteps < 1
        throw(ArgumentError("ArcLengthContinuation `maxsteps` must be ≥ 1, got $maxsteps"))
    end
    return ArcLengthContinuation(
        inner, initial_step_factor, adaptive, min_ds, max_step_factor,
        expand_factor, expand_threshold, max_angle, predictor, autodiff, maxsteps
    )
end

# Residual of the augmented (n+1) corrector system: the n homotopy equations stacked
# with the scalar Keller pseudo-arclength constraint. A named struct (not a closure) so
# the inner solver's compilation is reused across continuation steps. `f` is the raw
# user homotopy `f(u, p, λ)` / `f(du, u, p, λ)`; `τ` and `xcur` are the unit predictor
# direction and the last accepted packed point `[u; λ]`; `ds` is the arclength step. The
# augmented variable is `x = [u; λ]`; the solver passes the problem parameter `p` through
# (as for any `NonlinearProblem` residual) and it is forwarded to the user homotopy.
struct AugmentedHomotopy{F, V, T}
    f::F
    τ::V
    xcur::V
    ds::T
    n::Int
end

function (a::AugmentedHomotopy)(x, p)
    u = x[1:(a.n)]
    λ = x[a.n + 1]
    Hval = a.f(u, p, λ)
    c = LinearAlgebra.dot(a.τ, x .- a.xcur) - a.ds
    return vcat(Hval, c)
end

function (a::AugmentedHomotopy)(res, x, p)
    n = a.n
    a.f(view(res, 1:n), view(x, 1:n), p, x[n + 1])
    res[n + 1] = LinearAlgebra.dot(a.τ, x .- a.xcur) - a.ds
    return nothing
end

# The n homotopy rows on their own, as a function of the packed variable `x = [u; λ]`. The
# `:tangent` predictor differentiates this with NonlinearSolve's own Jacobian tooling to
# get the augmented Jacobian `[∂H/∂u | ∂H/∂λ]`; unlike `AugmentedHomotopy` it omits the
# arclength-constraint row (that row is a known constant, not part of the path Jacobian).
struct HomotopyResidual{F}
    f::F
    n::Int
end

(a::HomotopyResidual)(x, p) = a.f(view(x, 1:(a.n)), p, x[a.n + 1])

function (a::HomotopyResidual)(res, x, p)
    a.f(res, view(x, 1:(a.n)), p, x[a.n + 1])
    return nothing
end

# Natural-parameter solve at a fixed λ: gets the start point onto the curve and lands the
# final point exactly on λ = λspan[2]. Mirrors HomotopySweep's per-step solve.
function _arclength_fixed_solve(
        prob::SciMLBase.HomotopyProblem{uType, iip}, inner, uguess,
        λfix, args...; kwargs...
    ) where {uType, iip}
    fλ = SciMLBase.NonlinearFunction{iip}(FixLambda(prob.f, λfix))
    inner_prob = NonlinearProblem{iip}(fλ, copy(uguess), prob.p)
    return solve(inner_prob, inner, args...; prob.kwargs..., kwargs...)
end

# Build the reusable Jacobian cache for the `:tangent` predictor's augmented Jacobian,
# using NonlinearSolve's own `construct_jacobian_cache` (backend selection, tag handling,
# and AD-extras preparation shared with every other solver) rather than a bespoke DI call.
# `alg` carries no `concrete_jac`, so the cache holds a dense J that `nullspace` can use.
function _arclength_jac_cache(
        prob::SciMLBase.HomotopyProblem{uType, iip}, alg, x,
        n
    ) where {uType, iip}
    gf = SciMLBase.NonlinearFunction{iip}(HomotopyResidual(prob.f, n))
    gprob = NonlinearProblem{iip}(gf, x, prob.p)
    fu = if iip
        res = Utils.safe_similar(x, n)
        gf(res, x, prob.p)
        res
    else
        gf(x, prob.p)
    end
    autodiff = select_jacobian_autodiff(gprob, alg.autodiff)
    return construct_jacobian_cache(
        gprob, alg, gf, fu, x, prob.p; stats = NLStats(0, 0, 0, 0, 0), autodiff
    )
end

# Unit tangent of the curve H(u, λ) = 0 at the packed point `x = [u; λ]`, the null vector
# of the augmented Jacobian `J = [∂H/∂u | ∂H/∂λ]` (n×(n+1)) evaluated by `jac_cache`. `J`
# has full row rank n along the whole path — including at a fold, where the u-block ∂H/∂u
# is singular but the appended ∂H/∂λ column keeps the row rank — so `nullspace(J)` is
# one-dimensional and yields the tangent (vertical in λ at a fold) without needing a
# bordering vector. Oriented to continue the previous direction `τprev` (dot > 0).
function _arclength_tangent(jac_cache, x, τprev)
    J = jac_cache(x)
    N = LinearAlgebra.nullspace(J)
    τ = N[:, 1]
    τ ./= norm(τ)
    LinearAlgebra.dot(τ, τprev) < 0 && (τ .= .-τ)
    return τ
end

function CommonSolve.solve(
        prob::SciMLBase.HomotopyProblem{uType, iip},
        alg::ArcLengthContinuation, args...; kwargs...
    ) where {uType, iip}
    λ0, λ1 = prob.λspan
    λ = float(λ0)
    λT = typeof(λ)
    λend = λT(λ1)
    span = λend - λ

    # Correct the start onto the curve at λ0; everything downstream assumes H(u, λ) = 0.
    start_sol = _arclength_fixed_solve(prob, alg.inner, prob.u0, λ, args...; kwargs...)
    if !SciMLBase.successful_retcode(start_sol)
        return SciMLBase.build_solution(
            prob, alg, copy(prob.u0), start_sol.resid;
            retcode = start_sol.retcode, original = start_sol
        )
    end
    u = copy(start_sol.u)
    last_sol = start_sol
    if span == 0
        return SciMLBase.build_solution(
            prob, alg, u, last_sol.resid; retcode = ReturnCode.Success
        )
    end

    n = length(u)
    Tx = promote_type(eltype(u), λT)
    pack(uvec, λval) = Tx[uvec..., λval]
    xcur = pack(u, λ)
    xprev = xcur                       # no history yet → secant falls back to pure-λ
    have_prev = false

    min_ds = alg.min_ds === nothing ? sqrt(eps(λT)) : λT(alg.min_ds)
    max_ds = λT(alg.max_step_factor) * abs(span)
    ds = min(λT(alg.initial_step_factor) * abs(span), max_ds)
    sλ = λT(sign(span))
    cos_reject = Tx(cos(alg.max_angle))         # turn beyond max_angle ⇒ reject + shrink
    cos_grow = Tx(cos(alg.max_angle / 3))       # turn below max_angle/3 ⇒ allow growth
    streak = 0

    use_tangent = alg.predictor === :tangent
    # one reusable Jacobian cache (via NonlinearSolve's own tooling) for the tangent
    # predictor; nothing for the derivative-free secant.
    jac_cache = use_tangent ? _arclength_jac_cache(prob, alg, xcur, n) : nothing
    # orientation reference for the tangent predictor: seed toward λend so the first
    # tangent continues into the span (pure-λ direction picks the correct sign).
    τ = pack(zeros(Tx, n), sλ)

    for _ in 1:(alg.maxsteps)
        # Predictor direction τ (unit, length n+1).
        if use_tangent
            # True path tangent (null vector of the augmented Jacobian), oriented to
            # continue τ; accurate from the first step and well-defined at a fold.
            τ = _arclength_tangent(jac_cache, xcur, τ)
        elseif have_prev
            # Secant through the last two accepted points.
            d = xcur .- xprev
            dnorm = norm(d)
            τ = dnorm > 0 ? d ./ dnorm : pack(zeros(Tx, n), sλ)
        else
            # Bootstrap: a pure-λ step toward λend (no history for a secant yet).
            τ = pack(zeros(Tx, n), sλ)
        end

        xpred = xcur .+ ds .* τ
        aug = AugmentedHomotopy(prob.f, τ, xcur, Tx(ds), n)
        # length n+1; never in-place even for an iip homotopy — the constraint row has no
        # user-facing buffer, so we always own the residual.
        augf = SciMLBase.NonlinearFunction{iip}(aug)
        corr_prob = NonlinearProblem{iip}(augf, copy(xpred), prob.p)
        last_sol = solve(corr_prob, alg.inner, args...; prob.kwargs..., kwargs...)

        if SciMLBase.successful_retcode(last_sol)
            xnew = last_sol.u
            chord = xnew .- xcur
            nchord = norm(chord)

            # Curvature control: the realized step direction vs. the predictor measures the
            # path's turn. A large turn means either real high curvature or that the
            # corrector jumped to another branch — both call for a smaller step, so reject
            # and bisect. The gate needs a trustworthy predictor: the tangent is accurate
            # from the first step, but the secant only becomes meaningful once there is
            # history (its pure-λ bootstrap is legitimately misaligned with a sloped branch).
            trust = use_tangent || have_prev
            cosang = (trust && nchord > 0) ?
                clamp(LinearAlgebra.dot(τ, chord) / nchord, -one(Tx), one(Tx)) : one(Tx)
            if trust && cosang < cos_reject && alg.adaptive && ds / 2 >= min_ds
                ds = ds / 2
                streak = 0
                continue
            end

            unew = xnew[1:n]
            λnew = xnew[n + 1]
            λold = λ

            xprev = xcur
            xcur = copy(xnew)
            u = copy(unew)
            λ = λnew
            have_prev = true

            # A step that brackets λend has crossed the target; land on it exactly with a
            # λ-fixed correction warm-started by interpolation along the just-taken step.
            if (λold - λend) * (λnew - λend) <= 0
                denom = λnew - λold
                frac = denom == 0 ? one(Tx) : Tx((λend - λold) / denom)
                uprev = xprev[1:n]
                uguess = uprev .+ frac .* (unew .- uprev)
                final_sol = _arclength_fixed_solve(
                    prob, alg.inner, uguess, λend, args...; kwargs...
                )
                if SciMLBase.successful_retcode(final_sol)
                    return SciMLBase.build_solution(
                        prob, alg, copy(final_sol.u), final_sol.resid;
                        retcode = ReturnCode.Success
                    )
                end
                # the interpolation guess missed the basin; keep tracking and try the
                # next crossing rather than failing on a single bad warm start
            end

            # Grow only on near-straight stretches, so the step does not race ahead into a
            # turn it would then have to reject.
            if alg.adaptive
                streak += 1
                if streak >= alg.expand_threshold && cosang >= cos_grow
                    ds = min(λT(alg.expand_factor) * ds, max_ds)
                    streak = 0
                end
            end
        elseif alg.adaptive && ds / 2 >= min_ds
            ds = ds / 2          # bisect; retry from the same accepted point
            streak = 0
        else
            return SciMLBase.build_solution(
                prob, alg, u, nothing;
                retcode = last_sol.retcode, original = last_sol
            )
        end
    end

    # Ran out of attempts without bracketing λend: the path never reached the target.
    return SciMLBase.build_solution(prob, alg, u, nothing; retcode = ReturnCode.MaxIters)
end
