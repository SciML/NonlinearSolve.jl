"""
    HomotopyPolyAlgorithm(algs::Tuple; warm_handoff = true, store_original = Val(false))
    HomotopyPolyAlgorithm(; warm_handoff = true, store_original = Val(false))

A polyalgorithm for [`SciMLBase.HomotopyProblem`](@ref): a container for a tuple of
continuation algorithms that are tried in order until one returns a solution with a
successful retcode. The first success is returned immediately — later stages never run.
If every stage fails, the *last* stage's failed solution is returned, so its `retcode`
(and `original`, when the stage attaches one) describe the most robust attempt.

This is the default algorithm for `SciMLBase.HomotopyProblem`: `solve(prob)` and
`solve(prob, nothing)` route here.

The zero-argument form defaults to

```julia
HomotopyPolyAlgorithm((HomotopySweep(), ArcLengthContinuation()))
```

which encodes the natural escalation for homotopy solves: [`HomotopySweep`](@ref) is the
cheap first attempt — natural-parameter continuation marches the scalar ``λ``
monotonically across `λspan`, reusing one inner-solver cache across all steps, but it can
never reverse ``λ`` and therefore cannot follow a solution branch around a *fold*
(turning point). When the sweep fails, [`ArcLengthContinuation`](@ref) takes over: it
tracks the curve by pseudo-arclength in the augmented ``(u, λ)`` space, so ``λ`` is free
to decrease along the path and folds that defeat the sweep are rounded — at the higher
cost of solving an ``(n+1)``-dimensional corrector system per step.

### Warm handoff

When a [`HomotopySweep`](@ref) stage fails *partway* along the span, everything it
accepted before the failure is genuine converged path: its last accepted iterate is a
solution of ``H(u, λ) = 0`` at some ``λ`` strictly between `λspan[1]` and the failure
point. With `warm_handoff = true` (the default), the next stage is first attempted on
the remaining stretch — the problem is rebuilt with `u0` set to that last accepted
iterate and `λspan` shrunk to `(λ_h, λspan[2])` — instead of redoing the
already-conquered prefix from a cold start at `λspan[1]`.

The handoff λ is deliberately *backed off* from the failure: `λ_h` is placed 5% of the
span width behind the sweep's last accepted ``λ``. The sweep typically dies at a fold,
where the path turns vertical in ``λ``; a warm stage seeded right at the fold starts
with its initial pure-λ tangent nearly orthogonal to the true path direction and pays
for it in rejected steps (measured on the cubic S-curve: arclength warm-started at the
fold costs *more* residual calls than a full cold run, while backing off 5% costs
~15–25% less). The handed-over `u0` is the last accepted iterate — off-path at `λ_h`
by the backoff distance — and the stage's own λ-fixed anchor solve at `λ_h` pulls it
back onto the path for a few warm Newton iterations. Because the stages measure their
step-size *caps* (`max_step_factor`, and a sweep's fixed-size `nsteps`) as fractions
of the span width, the warm attempt rescales those caps by
`full_width / remaining_width` (capping the fraction at 1, i.e. at an absolute step
of the remaining width) so a user-tightened absolute cap survives the span shrink —
the initial step factor is left span-relative, since starting small right behind the
fold is measurably cheaper. Should the warm-started attempt
fail anyway, the stage is retried cold on the original full-range problem before the
polyalgorithm moves on, so enabling the handoff never costs robustness relative to
`warm_handoff = false` — only, in that rare double-failure case, the extra warm
attempt.

The handoff only engages when the sweep made real progress (the backed-off `λ_h` lies
strictly past `λspan[1]`); a sweep that failed at the `λspan[1]` anchor itself, or
within the backoff width of it, leaves the fallback stages with the current cold
full-range behavior. A warm-handoff success is returned as a solution of the
*original* problem (same `prob`, same `u` type); the stage's solution of the shrunken
problem is attached as `original` only when `store_original = Val(true)` (see below) —
by default it is dropped so the returned solution stays concretely typed.

### Arguments

  - `algs`: a tuple of continuation algorithms to try in order. Each stage must support
    `solve(prob::SciMLBase.HomotopyProblem, alg, args...; kwargs...)`.

### Keyword Arguments

  - `warm_handoff`: when `true` (default), a stage following a partway-failed
    [`HomotopySweep`](@ref) first attempts the remaining `(λ_h, λspan[2])` stretch
    from the sweep's last accepted iterate (with `λ_h` backed off 5% of the span from
    the failure), falling back to the cold full-range attempt only if that fails.
    `false` recovers the plain try-each-stage-cold behavior.
  - `store_original`: whether a warm-handoff success stores its shrunken-problem stage
    solution in the returned solution's `original` field. Default `Val(false)` keeps the
    returned solution concretely typed (the stage solution the handoff produces infers
    as `Any`). Pass `Val(true)` to recover the stage solution through `original` for
    introspection, at the cost of the returned solution's type no longer being concrete.

### Example

```julia
using NonlinearSolve

alg = HomotopyPolyAlgorithm() # HomotopySweep, then ArcLengthContinuation on failure
alg = HomotopyPolyAlgorithm((
    HomotopySweep(; inner = NewtonRaphson()),
    ArcLengthContinuation(; predictor = :tangent),
))
alg = HomotopyPolyAlgorithm(; warm_handoff = false) # always restart stages cold
```
"""
@concrete struct HomotopyPolyAlgorithm <: AbstractNonlinearSolveAlgorithm
    algs <: Tuple
    warm_handoff::Bool
    store_original <: Val
end

function HomotopyPolyAlgorithm(
        algs::Tuple; warm_handoff::Bool = true, store_original::Val = Val(false)
    )
    return HomotopyPolyAlgorithm(algs, warm_handoff, store_original)
end

function HomotopyPolyAlgorithm(;
        warm_handoff::Bool = true, store_original::Val = Val(false)
    )
    return HomotopyPolyAlgorithm(
        (HomotopySweep(), ArcLengthContinuation()); warm_handoff, store_original
    )
end

# Step-size *caps* of the bundled continuation stages are fractions of the λspan
# width by contract, so handing a stage the shrunken remaining span silently shrinks
# its absolute maximum step by the same ratio — crippling exactly the runs where the
# user tightened `max_step_factor` (measured on the cubic S-curve with
# `max_step_factor = 0.05`: the warm stage costs 2.4× a cold full-range run when its
# cap shrinks with the span, and beats it once the cap is rescaled). Rescaling by
# `scale = full_width / remaining_width` preserves the absolute cap the full-range
# run was already allowed — but only up to a fraction of 1 (an absolute cap of the
# remaining width): the warm stage starts and finishes in the curviest stretch of the
# path, and absolute caps beyond the remaining width measurably cost rejected steps
# there (S-curve, default `max_step_factor = 1.0`: warm stage 166 calls with the cap
# rescaled unboundedly vs 146 capped at the remaining width). The *initial* step
# factor is NOT rescaled for the same reason — starting small right behind the fold
# is measurably cheaper, and adaptive growth recovers the size within a few accepted
# steps. A sweep stage's fixed-size `nsteps` is rescaled like the cap: `nsteps` of
# the *remaining* span would shrink the absolute increment (and `adaptive = false`
# sweeps cannot recover from that). Stages of unknown type get the shrunken problem
# unchanged — no field contract to rescale against.
_rescale_step_caps(stage, scale) = stage
function _rescale_step_caps(stage::ArcLengthContinuation, scale)
    msf = stage.max_step_factor
    return @set stage.max_step_factor = min(msf * scale, oneunit(msf))
end
function _rescale_step_caps(stage::HomotopySweep, scale)
    if stage.nsteps !== nothing
        stage = @set stage.nsteps = max(1, ceil(Int, stage.nsteps / scale))
    end
    msf = stage.max_step_factor
    return @set stage.max_step_factor = min(msf * scale, oneunit(msf))
end
function _rescale_step_caps(stage::KantorovichHomotopy, scale)
    if stage.nsteps !== nothing
        stage = @set stage.nsteps = max(1, ceil(Int, stage.nsteps / scale))
    end
    msf = stage.max_step_factor
    return @set stage.max_step_factor = min(msf * scale, oneunit(msf))
end

function CommonSolve.solve(
        prob::SciMLBase.HomotopyProblem{uType, iip}, alg::HomotopyPolyAlgorithm,
        args...; kwargs...
    ) where {uType, iip}
    isempty(alg.algs) &&
        throw(ArgumentError("HomotopyPolyAlgorithm requires at least one algorithm"))
    nstages = length(alg.algs)
    λ0, λ1 = prob.λspan
    # Handoff seed `(u_last, λ_h)` from the most recent partway-failed sweep stage,
    # or `nothing` when none is available: `u_last` is the sweep's last accepted
    # iterate and `λ_h` its λ backed off by 5% of the span (see the loop below).
    handoff = nothing
    for (i, stage) in enumerate(alg.algs)
        if alg.warm_handoff && handoff !== nothing
            u_h, λ_h = handoff
            hprob = SciMLBase.HomotopyProblem{iip}(
                prob.f, u_h, prob.p; λspan = (λ_h, oftype(λ_h, λ1)), prob.kwargs...
            )
            scale = abs(oftype(λ_h, λ1) - oftype(λ_h, λ0)) / abs(oftype(λ_h, λ1) - λ_h)
            hsol = CommonSolve.solve(
                hprob, _rescale_step_caps(stage, scale), args...; kwargs...
            )
            if SciMLBase.successful_retcode(hsol)
                # Rebuild against the ORIGINAL problem: the caller handed in `prob`
                # and must get a solution whose `prob`/λspan semantics match it. The
                # shrunken-problem solution stays reachable through `original` when
                # `store_original = Val(true)`; by default the slot is pinned to
                # `Nothing` so the warm-handoff solution stays concretely typed (the
                # inner stage solve infers as `Any`).
                return build_solution_less_specialize(
                    prob, stage, hsol.u, hsol.resid;
                    retcode = hsol.retcode, original = hsol, stats = hsol.stats,
                    store_original = alg.store_original
                )
            end
            # Warm attempt failed (even the backed-off seed can be unlucky, e.g. on a
            # path that is degenerate across the whole handoff neighborhood): fall
            # through to the cold full-range attempt so the handoff never costs
            # robustness relative to the plain staged behavior.
        end
        sol = if stage isa Union{HomotopySweep, KantorovichHomotopy}
            csol, λ_last = _homotopy_sweep_solve(prob, stage, args...; kwargs...)
            if λ_last !== nothing
                # The sweep dies where the path turns hard (a fold), and a fallback
                # seeded exactly there starts with its initial pure-λ tangent nearly
                # orthogonal to the path — measured on the cubic S-curve, arclength
                # warm-started AT the fold costs more residual calls than a full
                # cold run, while 5% of the span behind it costs ~15–25% less. So
                # the handoff λ is backed off by span/20 from the last accepted
                # point; the stage's own λ-fixed anchor solve pulls `u_last` (5% of
                # the span away in λ) back onto the path at `λ_h` for a few
                # Newton iterations. The handoff only engages when the backed-off λ
                # is still strictly past `λspan[1]` — a sweep that died within the
                # backoff width of the anchor leaves the fallback no cheaper start
                # than its own anchor solve.
                backoff = (oftype(λ_last, λ1) - oftype(λ_last, λ0)) / 20
                if abs(λ_last - oftype(λ_last, λ0)) > abs(backoff)
                    handoff = (csol.u, λ_last - backoff)
                end
            end
            csol
        else
            CommonSolve.solve(prob, stage, args...; kwargs...)
        end
        (SciMLBase.successful_retcode(sol) || i == nstages) && return sol
    end
    return
end
