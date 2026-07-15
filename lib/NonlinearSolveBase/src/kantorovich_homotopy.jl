@doc raw"""
    KantorovichHomotopy(; inner = nothing, nsteps = nothing,
        initial_step_factor = 0.1, min_dλ = nothing, max_step_factor = 1.0,
        qmin = 1 // 5, qmax = 5, Θmin = 1 // 8, Θreject = 0.95,
        Θbar = 0.5, γ = 0.95, strict = true, predictor = :constant,
        predictor_order = nothing, expand_quality = 0.25,
        tracking_maxiters = 10, tracking_abstol = nothing, maxsteps = 10000,
        store_original = Val(false))

Natural-parameter continuation for a [`SciMLBase.HomotopyProblem`](@ref), with step
sizes chosen from the observed contraction of the inner nonlinear corrector. This is
the Newton--Kantorovich path-following controller described in Section 5.1.3 of
Deuflhard, *Newton Methods for Nonlinear Problems*.

For each accepted continuation point, the solver measures residual contraction ratios

```math
\Theta_k = \frac{\lVert H(u^{k + 1}, \lambda)\rVert}
                 {\lVert H(u^k, \lambda)\rVert}
```

during the corrector. If ``\Theta_0`` is the first available ratio, the next parameter
increment is multiplied by

```math
q = \operatorname{clamp}\left(
    \gamma \left[\frac{g(\bar\Theta)}{g(\max(\Theta_0, \Theta_{min}))}\right]^{1/p},
    q_{min}, q_{max}\right), \qquad g(x) = \sqrt{1 + 4x} - 1.
```

Here ``p`` is the predictor order: by default `1` for `predictor = :constant` and `2`
for `predictor = :secant`. The constant predictor matches `ImplicitDiscreteSolve`'s
controller and is the default. A corrector whose contraction exceeds `Θreject` is rejected
when `strict = true`, even if the inner solver eventually converged, and is retried with
the smaller increment prescribed by the same formula.

The driver shares the cache reuse, secant trust monitoring, prediction-quality growth
gate, interior iteration cap, optional loose tracking tolerance, analytic/sparse
Jacobian forwarding, and endpoint polishing of [`HomotopySweep`](@ref). Compared with
`HomotopySweep`, only the parameter-step controller differs: `HomotopySweep` uses
success streaks and coarse corrector-effort bands, while `KantorovichHomotopy` uses the
measured contraction rate after every corrector.

Keyword arguments:

  - `inner`: the inner nonlinear algorithm. `nothing` selects NonlinearSolve's default
    polyalgorithm. A polyalgorithm contains separate corrector sequences whose residual
    contractions are not comparable across rungs, and algorithms without an iterative
    cache do not expose intermediate residuals, so those solves use `Θmin`. Pass a single
    cache-based iterative algorithm such as `NewtonRaphson()` to activate the
    measured-contraction controller.
  - `nsteps`: optional number of equal divisions used only to choose the initial
    parameter increment. Subsequent increments remain adaptive.
  - `initial_step_factor`: initial increment as a fraction of the `λspan` width when
    `nsteps` is not supplied.
  - `min_dλ`, `max_step_factor`: minimum absolute increment and maximum increment as a
    fraction of the span width. `min_dλ = nothing` resolves to
    `sqrt(eps(typeof(λ)))`.
  - `qmin`, `qmax`: lower and upper bounds for the step-size multiplier.
  - `Θmin`: contraction-rate floor used when a corrector converges before a ratio can be
    measured or when the measured contraction is smaller than the floor.
  - `Θbar`: target corrector contraction rate.
  - `Θreject`: contraction rate above which a corrector is rejected in strict mode.
  - `γ`: safety factor in the Kantorovich step-size formula.
  - `strict`: reject converged correctors containing any contraction ratio greater than
    `Θreject`. With `false`, every converged corrector is accepted and the ratio only
    controls the following increment.
  - `predictor`: `:secant` or `:constant`, with the same trust-monitored behavior as
    [`HomotopySweep`](@ref).
  - `predictor_order`: exponent denominator ``p`` in the controller formula. `nothing`
    selects `2` for the secant predictor and `1` for the constant predictor.
  - `expand_quality`: maximum relative secant-prediction error that permits `q > 1`.
    A corrector taking at most two iterations also permits growth. `Inf` disables this
    gate.
  - `tracking_maxiters`, `tracking_abstol`, `maxsteps`, `store_original`: identical to
    the corresponding [`HomotopySweep`](@ref) options.

This algorithm follows `λ` monotonically and therefore cannot round a fold. Combine it
with [`ArcLengthContinuation`](@ref) in a [`HomotopyPolyAlgorithm`](@ref) when a fold
must be traversed.
"""
@concrete struct KantorovichHomotopy <: AbstractNonlinearSolveAlgorithm
    inner
    nsteps
    initial_step_factor
    min_dλ
    max_step_factor
    qmin
    qmax
    Θmin
    Θreject
    Θbar
    γ
    strict::Bool
    predictor::Symbol
    predictor_order::Int
    expand_quality
    tracking_maxiters
    tracking_abstol
    maxsteps::Int
    store_original <: Val
end

function KantorovichHomotopy(;
        inner = nothing, nsteps = nothing, initial_step_factor = 0.1,
        min_dλ = nothing, max_step_factor = 1.0, qmin = 1 // 5, qmax = 5,
        Θmin = 1 // 8, Θreject = 0.95, Θbar = 0.5, γ = 0.95,
        strict = true, predictor = :constant, predictor_order = nothing,
        expand_quality = 0.25, tracking_maxiters = 10, tracking_abstol = nothing,
        maxsteps = 10000, store_original::Val = Val(false)
    )
    nsteps !== nothing && nsteps < 1 &&
        throw(ArgumentError("KantorovichHomotopy `nsteps` must be ≥ 1, got $nsteps"))
    !(0 < initial_step_factor <= 1) && throw(
        ArgumentError(
            "KantorovichHomotopy `initial_step_factor` must be in (0, 1], " *
                "got $initial_step_factor"
        )
    )
    min_dλ !== nothing && min_dλ <= 0 && throw(
        ArgumentError("KantorovichHomotopy `min_dλ` must be positive, got $min_dλ")
    )
    !(0 < max_step_factor <= 1) && throw(
        ArgumentError(
            "KantorovichHomotopy `max_step_factor` must be in (0, 1], " *
                "got $max_step_factor"
        )
    )
    !(0 < qmin < 1) &&
        throw(ArgumentError("KantorovichHomotopy `qmin` must be in (0, 1), got $qmin"))
    qmax < 1 &&
        throw(ArgumentError("KantorovichHomotopy `qmax` must be ≥ 1, got $qmax"))
    !(0 < Θmin <= Θbar < Θreject < 1) && throw(
        ArgumentError(
            "KantorovichHomotopy requires 0 < Θmin ≤ Θbar < Θreject < 1; " *
                "got Θmin = $Θmin, Θbar = $Θbar, Θreject = $Θreject"
        )
    )
    !(0 < γ < 1) &&
        throw(ArgumentError("KantorovichHomotopy `γ` must be in (0, 1), got $γ"))
    predictor in (:secant, :constant) || throw(
        ArgumentError(
            "KantorovichHomotopy `predictor` must be :secant or :constant, " *
                "got :$predictor"
        )
    )
    predictor_order = something(predictor_order, predictor === :secant ? 2 : 1)
    predictor_order < 1 && throw(
        ArgumentError(
            "KantorovichHomotopy `predictor_order` must be ≥ 1, got $predictor_order"
        )
    )
    !(expand_quality > 0) && throw(
        ArgumentError(
            "KantorovichHomotopy `expand_quality` must be positive, got $expand_quality"
        )
    )
    tracking_maxiters !== nothing && tracking_maxiters < 1 && throw(
        ArgumentError(
            "KantorovichHomotopy `tracking_maxiters` must be ≥ 1, got " *
                "$tracking_maxiters"
        )
    )
    tracking_abstol !== nothing && !(tracking_abstol > 0) && throw(
        ArgumentError(
            "KantorovichHomotopy `tracking_abstol` must be positive, got " *
                "$tracking_abstol"
        )
    )
    maxsteps < 1 &&
        throw(ArgumentError("KantorovichHomotopy `maxsteps` must be ≥ 1, got $maxsteps"))

    return KantorovichHomotopy(
        inner, nsteps, initial_step_factor, min_dλ, max_step_factor, qmin, qmax,
        Θmin, Θreject, Θbar, γ, strict, predictor, predictor_order,
        expand_quality, tracking_maxiters, tracking_abstol, maxsteps, store_original
    )
end
