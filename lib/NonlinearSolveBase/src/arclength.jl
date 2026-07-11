"""
    ArcLengthContinuation(; inner = nothing, initial_step_factor = 0.1,
        adaptive = true, min_ds = nothing, max_step_factor = 1.0,
        expand_factor = 2.0, expand_threshold = 2, max_angle = π / 6,
        predictor = :secant, autodiff = nothing, linsolve = nothing,
        tracking_maxiters = 10, maxsteps = 10000, theta = 0.5,
        store_original = Val(false))

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

Optional derivative fields of the problem's `NonlinearFunction` (which follow the same
λ-extended argument convention as the residual) are consumed. An analytic
`jac(u, p, λ)` / `jac(J, u, p, λ)` supplies the ``∂H/∂u`` block of the augmented path
Jacobian ``[∂H/∂u | ∂H/∂λ]``; the missing ``∂H/∂λ`` column is a *scalar-parameter*
derivative, obtained as one forward-mode derivative of the residual in ``λ`` at fixed
`u` (through `autodiff`) — the packed system is never differentiated wholesale. The
assembled path Jacobian drives the `:tangent` predictor and, extended by the
analytically known θ-weighted Keller constraint row, gives every augmented corrector
solve a full analytic ``(n+1)×(n+1)`` Jacobian; the λ-fixed anchor and landing solves
consume the jac exactly as [`HomotopySweep`](@ref) does. A `jac_prototype` (or matrix
`sparsity`) is extended to the augmented shapes: one structurally dense ``∂H/∂λ``
column for the predictor's ``n×(n+1)`` system, plus the structurally dense constraint
row for the corrector's bordered ``(n+1)×(n+1)`` system. Sparse and structured
prototypes are promoted to `SparseMatrixCSC` — a bordered `Tridiagonal` is no longer
tridiagonal, and CSC is the general container the coloring and sparse linear-solve
machinery handle; this requires `SparseArrays` to be loaded (structured prototypes
fall back to a dense bordered prototype otherwise), and the prototype is not
eltype-promoted with λ. A sparsity *detector* is forwarded unchanged (it detects the
augmented pattern from the augmented residual itself). A user `colorvec` is forwarded
to the predictor's system extended by one fresh color for the dense λ column; it is
not forwarded to the corrector, whose dense constraint row admits no nontrivial column
coloring — there the bordered prototype's value is chiefly the sparse linear solve.
When no derivative fields are present, construction is identical to before.

!!! note "The arclength metric is θ-weighted"

    All of the solver's path geometry — the Keller constraint row, the predictor
    normalization and orientation, the realized chord length, and the bend-angle test —
    is measured in the weighted inner product

    ```
    ⟨(u₁, λ₁), (u₂, λ₂)⟩_θ = (θ/n)⋅⟨u₁, u₂⟩ + (1 - θ)⋅λ₁λ₂,       n = length(u)
    ```

    (the `DotTheta` convention of BifurcationKit.jl). The `1/n` normalization makes the
    balance between the state block and the parameter independent of the system size: in
    the plain Euclidean dot on `[u; λ]` the ``n`` state components swamp the single ``λ``
    component for large systems, distorting the constraint and the angle test. A
    consequence is that `ds` values (`initial_step_factor`, `min_ds`, `max_step_factor`)
    and `max_angle` are measured in this weighted metric, **not** in the Euclidean metric
    on `[u; λ]`: for a pure-``λ`` motion a weighted arclength `ds` corresponds to a
    ``λ``-distance of `ds / sqrt(1 - θ)`, and for a pure-``u`` motion to a Euclidean
    ``u``-distance of `ds / sqrt(θ/n)`. Step-size values tuned against versions that used
    the unweighted metric may need rescaling.

Keyword arguments:

  - `inner`: the inner nonlinear algorithm used for both the initial on-curve correction
    and the augmented corrector; `nothing` selects NonlinearSolve's default polyalgorithm.
    A polyalgorithm inner runs the augmented corrector with best-subalgorithm retention
    (see [`NonlinearSolvePolyAlgorithm`](@ref)): after the first corrector solve
    discovers the winning subalgorithm, each warm-started corrector resumes from it
    instead of re-running the ladder from its start, escalating only when it fails. The
    λ-fixed anchor and landing solves always run the full ladder.
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
    fold (where the tangent is vertical in ``λ``). It is obtained from the bordered
    linear solve ``[J; τ_prevᵀ] t = e_{n+1}`` (one LU factorization per step, sparse
    Jacobians supported), falling back to a dense SVD null-space computation only when
    the bordered matrix is (near-)singular — e.g. exactly at a branch point, or when the
    previous tangent is orthogonal to the path. The tangent is a higher-order predictor
    and is accurate from the first step, so it curvature-checks every step and can round a
    fold at the very start; the cost is one Jacobian factorization per step (see
    `autodiff`). This is the Euler tangent predictor of the classic path trackers and of
    OpenModelica's global homotopy.
  - `autodiff`: the automatic-differentiation backend (an `ADTypes.AbstractADType`) used
    to form the augmented Jacobian for the `:tangent` predictor and, when the problem
    supplies an analytic `jac`, to take the single ``∂H/∂λ`` scalar derivative that
    completes it; `nothing` (default) selects `AutoForwardDiff()`. Unused by the
    `:secant` predictor on problems without an analytic `jac`.
  - `linsolve`: the LinearSolve.jl algorithm for the `:tangent` predictor's bordered
    solve (the same knob the Newton descent methods expose); `nothing` (default) selects
    LinearSolve's default. Unused by the `:secant` predictor.
  - `tracking_maxiters`: iteration cap for the augmented corrector solves (default 10,
    in the range used by MatCont, HomotopyContinuation.jl, and OpenModelica;
    `nothing` disables). A rejected step retries at half the arclength increment from a warm
    start, so failing fast is far cheaper than exhausting the inner solver's full
    budget. Never applied to the anchor or final λ-fixed landing solves; an explicit
    user-passed `maxiters` always wins. On success, step growth is additionally
    scaled by the corrector's iteration count (AUTO-style bands) alongside the
    bend-angle gate.
  - `maxsteps`: a hard cap on the total number of predictor-corrector attempts (including
    bisection retries). Required because the path is *not* monotone in `λ`, so a sweep
    that never reaches the target — a closed loop, or a branch escaping to infinity —
    would otherwise not terminate. Exceeding it returns a `ReturnCode.MaxIters` failure.
  - `theta`: the weight ``θ`` of the arclength metric (see the note above). Must be in
    `(0, 1)`; the default `0.5` weighs the (size-normalized) state block and the
    parameter equally. Larger `theta` emphasizes the state components, smaller `theta`
    emphasizes ``λ``.
  - `store_original`: whether to store the failing inner solve in the `original` field of
    the returned solution. Default `Val(false)` to keep the returned solution type
    concrete (the anchor/corrector inner solves' return type inference gives up, so
    storing one would pin the returned solution's `original` slot to `Any`). Set to
    `Val(true)` to keep the payload for debugging. Mirrors
    [`NonlinearSolvePolyAlgorithm`](@ref)'s option of the same name.

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
    linsolve
    tracking_maxiters
    maxsteps::Int
    theta
    store_original <: Val
end

function ArcLengthContinuation(;
        inner = nothing, initial_step_factor = 0.1, adaptive = true,
        min_ds = nothing, max_step_factor = 1.0, expand_factor = 2.0,
        expand_threshold = 2, max_angle = π / 6, predictor = :secant,
        autodiff = nothing, linsolve = nothing, tracking_maxiters = 10,
        maxsteps = 10000, theta = 0.5, store_original::Val = Val(false)
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
    if tracking_maxiters !== nothing && tracking_maxiters < 1
        throw(
            ArgumentError(
                "ArcLengthContinuation `tracking_maxiters` must be ≥ 1 (nothing " *
                    "disables the cap), got $tracking_maxiters"
            )
        )
    end
    if maxsteps < 1
        throw(ArgumentError("ArcLengthContinuation `maxsteps` must be ≥ 1, got $maxsteps"))
    end
    if !(0 < theta < 1)
        throw(
            ArgumentError(
                "ArcLengthContinuation `theta` must be in (0, 1), got $theta"
            )
        )
    end
    return ArcLengthContinuation(
        inner, initial_step_factor, adaptive, min_ds, max_step_factor,
        expand_factor, expand_threshold, max_angle, predictor, autodiff, linsolve,
        tracking_maxiters, maxsteps, theta, store_original
    )
end

# The θ-weighted inner product on packed points x = [u; λ] (BifurcationKit's `DotTheta`):
# ⟨x, y⟩_θ = wu⋅⟨x[1:n], y[1:n]⟩ + wλ⋅x[n+1]y[n+1] with wu = θ/n, wλ = 1 - θ. All of the
# driver's path geometry (constraint row, tangent/secant normalization, orientation,
# chord length, bend angle) must use this one metric consistently or the pieces measure
# different things. Written on views so it neither allocates nor forces the eltype —
# `x`/`y` may carry ForwardDiff duals when the corrector differentiates the constraint.
@inline function _theta_dot(x, y, wu, wλ, n)
    return wu * LinearAlgebra.dot(view(x, 1:n), view(y, 1:n)) + wλ * (x[n + 1] * y[n + 1])
end

@inline _theta_norm(x, wu, wλ, n) = sqrt(_theta_dot(x, x, wu, wλ, n))

# ⟨τ, x - x0⟩_θ without materializing the difference. The corrector residual evaluates
# this once per inner iteration with `x` possibly Dual-typed under the inner solver's
# AD, so a temporary `x .- x0` would heap-allocate on every residual call; the loop
# stays generic in the eltype (duals flow through the accumulator).
@inline function _theta_dot_shifted(τ, x, x0, wu, wλ, n)
    acc = zero(τ[1] * (x[1] - x0[1]))
    @inbounds @simd for i in 1:n
        acc += τ[i] * (x[i] - x0[i])
    end
    return wu * acc + wλ * (τ[n + 1] * (x[n + 1] - x0[n + 1]))
end

# Residual of the augmented (n+1) corrector system: the n homotopy equations stacked
# with the scalar Keller pseudo-arclength constraint. A named struct (not a closure) so
# the inner solver's compilation is reused across continuation steps. `f` is the raw
# user homotopy `f(u, p, λ)` / `f(du, u, p, λ)`; `τ` and `xcur` are the (θ-metric unit)
# predictor direction and the last accepted packed point `[u; λ]` — both alias STABLE
# driver buffers that are updated in place between steps — and `ds` is a mutable field,
# so the SAME residual (and the one inner-solver cache built around it) is reused for
# every corrector attempt: advancing the continuation is a couple of buffer/field
# writes, not a new function/problem/solver allocation (mirroring the sweep's
# `FixLambda`). `wu`/`wλ` are the θ-metric weights, so the constraint row reads
# ⟨τ, x - xcur⟩_θ = ds. The augmented variable is `x = [u; λ]`; the solver passes the
# problem parameter `p` through (as for any `NonlinearProblem` residual) and it is
# forwarded to the user homotopy.
mutable struct AugmentedHomotopy{F, V, T}
    const f::F
    const τ::V
    const xcur::V
    ds::T
    const n::Int
    const wu::T
    const wλ::T
end

function (a::AugmentedHomotopy)(x, p)
    u = view(x, 1:(a.n))
    λ = x[a.n + 1]
    Hval = a.f(u, p, λ)
    c = _theta_dot_shifted(a.τ, x, a.xcur, a.wu, a.wλ, a.n) - a.ds
    return vcat(Hval, c)
end

function (a::AugmentedHomotopy)(res, x, p)
    n = a.n
    a.f(view(res, 1:n), view(x, 1:n), p, x[n + 1])
    res[n + 1] = _theta_dot_shifted(a.τ, x, a.xcur, a.wu, a.wλ, n) - a.ds
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

# The homotopy with λ moved into the differentiated-argument slot and (u, p) as DI
# `Constant` contexts: the ∂H/∂λ column of the augmented Jacobian is the derivative of
# this in its first (scalar) argument — one forward-mode directional derivative, not a
# Jacobian of the packed system.
struct LambdaShifted{F}
    f::F
end

(ls::LambdaShifted)(λ, u, p) = ls.f(u, p, λ)

function (ls::LambdaShifted)(res, λ, u, p)
    ls.f(res, u, p, λ)
    return nothing
end

# Analytic n×(n+1) path Jacobian `[∂H/∂u | ∂H/∂λ]` of the packed variable `x = [u; λ]`,
# assembled from the user's λ-extended `jac` (which supplies only the n×n ∂H/∂u block)
# plus the ∂H/∂λ column as a single scalar derivative in λ (prepared once at solve
# start, reused every call). Standard 2/3-argument Jacobian calling convention, so
# `construct_jacobian_cache` consumes it through its ordinary `has_jac` path and the
# augmented corrector's Jacobian can embed it as its top block.
struct AugmentedPathJac{J, L, B, P, R}
    jac::J           # the user's λ-extended jac(u, p, λ) / jac(J, u, p, λ)
    lres::L          # LambdaShifted residual for the ∂H/∂λ column
    backend::B
    prep::P
    rescratch::R     # iip: primal residual buffer for DI.derivative!; oop: nothing
    n::Int
end

function (apj::AugmentedPathJac)(J, x, p)
    n = apj.n
    u = view(x, 1:n)
    λ = x[n + 1]
    apj.jac(view(J, 1:n, 1:n), u, p, λ)
    DI.derivative!(
        apj.lres, apj.rescratch, view(J, 1:n, n + 1),
        apj.prep, apj.backend, λ, Constant(u), Constant(p)
    )
    return nothing
end

function (apj::AugmentedPathJac)(x, p)
    n = apj.n
    u = view(x, 1:n)
    λ = x[n + 1]
    Ju = apj.jac(u, p, λ)
    col = DI.derivative(apj.lres, apj.prep, apj.backend, λ, Constant(u), Constant(p))
    return hcat(Ju, col)
end

# Full analytic (n+1)×(n+1) Jacobian of `AugmentedHomotopy`: the path Jacobian as the
# top n rows and the Keller constraint row — analytically known, it is the θ-weighted
# τ — as the bottom row. Aliases the driver's in-place-updated τ buffer (the same
# object the per-step `AugmentedHomotopy` residual reads), so the Jacobian's constraint
# row can never desynchronize from the residual's; `ds`/`xcur` do not appear because
# the constraint is affine in `x` (its gradient is independent of both).
struct AugmentedHomotopyJac{PJ, V, T}
    pathjac::PJ
    τ::V
    n::Int
    wu::T
    wλ::T
end

function (aj::AugmentedHomotopyJac)(J, x, p)
    n = aj.n
    aj.pathjac(view(J, 1:n, :), x, p)
    @views J[n + 1, 1:n] .= aj.wu .* aj.τ[1:n]
    J[n + 1, n + 1] = aj.wλ * aj.τ[n + 1]
    return nothing
end

function (aj::AugmentedHomotopyJac)(x, p)
    n = aj.n
    Jtop = aj.pathjac(x, p)
    row = vcat(aj.wu .* view(aj.τ, 1:n), aj.wλ * aj.τ[n + 1])
    return vcat(Jtop, transpose(row))
end

# The user's n×n ∂H/∂u prototype extended to the packed path system's n×(n+1) shape by
# appending the structurally dense ∂H/∂λ column. Sparse/structured prototypes become
# `SparseMatrixCSC` — a bordered `Tridiagonal` (or any banded/structured type) no longer
# fits its own structure, and CSC is the general sparse container every downstream
# consumer (coloring, sparse LU) handles — via the STRUCTURAL nonzero pattern
# (`Utils.structural_sparse`), so band entries whose prototype values happen to be zero
# survive as pattern. The sparse route needs the SparseArrays extension; without it (a
# session that cannot construct CSC matrices anyway) structured prototypes fall back to
# dense. Dense prototypes stay dense. As in the sweep, the prototype is NOT
# eltype-promoted with λ.
function _augmented_prototype(proto::AbstractMatrix, n)
    T = eltype(proto)
    if sparse_or_structured_prototype(proto) &&
            Utils.is_extension_loaded(Val(:SparseArrays))
        S = Utils.structural_sparse(proto)
        return hcat(S, Utils.make_sparse(fill(one(T), n, 1)))
    end
    B = fill(one(T), n, n + 1)
    copyto!(view(B, 1:n, 1:n), proto)
    return B
end

# The (n+1)×(n+1) bordered prototype of the augmented corrector system: the augmented
# path prototype with the structurally dense Keller constraint row appended,
# `[∂H/∂u  ∂H/∂λ; wᵀ  1]`.
function _bordered_prototype(proto::AbstractMatrix, n)
    T = eltype(proto)
    top = _augmented_prototype(proto, n)
    row = fill(one(T), 1, n + 1)
    if sparse_or_structured_prototype(top)
        return vcat(top, Utils.make_sparse(row))
    end
    return vcat(top, row)
end

# Builds the shared analytic path Jacobian (or `nothing` when the user supplied no
# analytic jac; the branch is decided by the problem's type). The DI preparation for
# the ∂H/∂λ scalar derivative is done once here — with `(u, p)` as `Constant` contexts
# whose values change call to call — and reused by both consumers (the `:tangent`
# predictor's Jacobian cache and every corrector Jacobian evaluation).
function _arclength_path_jac(
        prob::SciMLBase.HomotopyProblem{uType, iip}, alg, x, n
    ) where {uType, iip}
    prob.f.jac === nothing && return nothing
    gf = SciMLBase.NonlinearFunction{iip}(HomotopyResidual(prob.f, n))
    gprob = NonlinearProblem{iip}(gf, x, prob.p)
    backend = select_jacobian_autodiff(gprob, alg.autodiff)
    lres = LambdaShifted(prob.f.f)
    u = view(x, 1:n)
    λ = x[n + 1]
    if iip
        rescratch = Utils.safe_similar(x, n)
        prep = DI.prepare_derivative(
            lres, rescratch, backend, λ, Constant(u), Constant(prob.p);
            strict = Val(false)
        )
        return AugmentedPathJac(prob.f.jac, lres, backend, prep, rescratch, n)
    else
        prep = DI.prepare_derivative(
            lres, backend, λ, Constant(u), Constant(prob.p); strict = Val(false)
        )
        return AugmentedPathJac(prob.f.jac, lres, backend, prep, nothing, n)
    end
end

# The packed path function for the `:tangent` predictor, carrying the derivative fields
# of the problem's NonlinearFunction translated to the packed n×(n+1) shapes: the
# analytic path Jacobian, the augmented prototype/matrix-sparsity, and the user
# colorvec extended by one fresh color for the structurally dense ∂H/∂λ column (valid
# for forward-mode column coloring, which is what the packed n×(n+1) system uses). A
# sparsity DETECTOR is forwarded unchanged — it detects the augmented pattern from the
# packed residual itself. With no derivative fields the construction is exactly the
# bare one (the branch is decided by the problem's type, so the unused arm is compiled
# away).
function _arclength_path_function(::Val{iip}, f, path_jac, n) where {iip}
    g = HomotopyResidual(f, n)
    if f.jac === nothing && f.jac_prototype === nothing && f.sparsity === nothing &&
            f.colorvec === nothing
        return SciMLBase.NonlinearFunction{iip}(g)
    end
    jac_prototype = f.jac_prototype === nothing ? nothing :
        _augmented_prototype(f.jac_prototype, n)
    # the NonlinearFunction constructor defaults `sparsity` to the SAME object as
    # `jac_prototype`, and construct_concrete_adtype rejects a matrix sparsity that is
    # a distinct object from a sparse/structured prototype — preserve the identity
    sparsity = if f.sparsity === nothing
        nothing
    elseif f.sparsity === f.jac_prototype
        jac_prototype
    elseif f.sparsity isa AbstractMatrix
        _augmented_prototype(f.sparsity, n)
    else
        f.sparsity
    end
    colorvec = f.colorvec === nothing ? nothing :
        vcat(f.colorvec, maximum(f.colorvec) + 1)
    return SciMLBase.NonlinearFunction{iip}(
        g; jac = path_jac, jac_prototype, sparsity, colorvec
    )
end

# The corrector's NonlinearFunction around the per-step `AugmentedHomotopy` residual,
# carrying the (n+1)×(n+1) bordered derivative fields. The user colorvec is NOT
# forwarded: the structurally dense constraint row makes every pair of columns share a
# row, so no nontrivial column coloring is valid for the bordered pattern — when sparse
# AD applies (no analytic jac), the coloring is recomputed, and the bordered
# prototype's value is chiefly the sparse linear solve. With no derivative fields the
# construction is exactly the bare one.
function _arclength_augmented_function(
        ::Val{iip}, f, aug, jac, jac_prototype, sparsity
    ) where {iip}
    if f.jac === nothing && f.jac_prototype === nothing && f.sparsity === nothing &&
            f.colorvec === nothing
        return SciMLBase.NonlinearFunction{iip}(aug)
    end
    return SciMLBase.NonlinearFunction{iip}(aug; jac, jac_prototype, sparsity)
end

# Natural-parameter solve at a fixed λ: gets the start point onto the curve and lands the
# final point exactly on λ = λspan[2]. Mirrors HomotopySweep's per-step solve, including
# its derivative-field forwarding (λ-fixed jac, prototype/sparsity/colorvec unchanged).
function _arclength_fixed_solve(
        prob::SciMLBase.HomotopyProblem{uType, iip}, inner, uguess,
        λfix, args...; kwargs...
    ) where {uType, iip}
    fλ = _sweep_nonlinear_function(Val(iip), prob.f, FixLambda(prob.f, λfix))
    inner_prob = NonlinearProblem{iip}(fλ, copy(uguess), prob.p)
    return solve(inner_prob, inner, args...; prob.kwargs..., kwargs...)
end

# Build the reusable Jacobian cache for the `:tangent` predictor's augmented Jacobian,
# using NonlinearSolve's own `construct_jacobian_cache` (backend selection, tag handling,
# and AD-extras preparation shared with every other solver) rather than a bespoke DI call.
# `alg` carries no `concrete_jac`, so the cache holds a concrete J the bordered solve can
# factorize.
function _arclength_jac_cache(
        prob::SciMLBase.HomotopyProblem{uType, iip}, alg, path_jac, x,
        n, stats
    ) where {uType, iip}
    gf = _arclength_path_function(Val(iip), prob.f, path_jac, n)
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
        gprob, alg, gf, fu, x, prob.p; stats, autodiff, linsolve = alg.linsolve
    )
end

# Workspace for the bordered tangent solve, built once per solve. `B` receives [J; wᵀ]
# each step, `rhs` holds the constant e_{n+1}, `t` receives the solution, and `lincache`
# is the same NonlinearSolve linear-solver cache the Newton descent methods use
# (LinearSolve.jl-backed, honoring the algorithm's `linsolve` and reusing the solver's
# workspace across steps; a plain `\`-backed native cache when the LinearSolve
# extension is not loaded, since NonlinearSolveBase only weak-depends on it).
@concrete struct BorderedTangentCache
    B
    rhs
    t
    r
    lincache
end

function _bordered_tangent_cache(alg, p, ::Type{Tx}, n, stats) where {Tx}
    B = zeros(Tx, n + 1, n + 1)
    rhs = zeros(Tx, n + 1)
    rhs[n + 1] = one(Tx)
    t = Vector{Tx}(undef, n + 1)
    r = Vector{Tx}(undef, n)
    linsolve = if alg.linsolve === nothing && !Utils.is_extension_loaded(Val(:LinearSolve))
        \
    else
        alg.linsolve
    end
    lincache = construct_linear_solver(alg, linsolve, B, rhs, t, p; stats)
    return BorderedTangentCache(B, rhs, t, r, lincache)
end

# θ-metric unit tangent of the curve H(u, λ) = 0 given the augmented Jacobian
# `J = [∂H/∂u | ∂H/∂λ]` (n×(n+1)). `J` has full row rank n along the whole path —
# including at a fold, where the u-block ∂H/∂u is singular but the appended ∂H/∂λ column
# keeps the row rank — so its null space is one-dimensional and is the tangent (vertical
# in λ at a fold). The null vector is extracted with the bordered solve
#     [J; wᵀ] t = e_{n+1},        w = θ-metric weighting of τprev,
# i.e. J t = 0 with the normalization ⟨τprev, t⟩_θ = 1 (Keller/Govaerts bordering;
# BifurcationKit's `Bordered` tangent) — one LU factorization, O(n³/3) dense and
# sparse-capable, versus the O(n³)-with-large-constant dense SVD of `nullspace`. The
# bordered matrix is singular exactly when τprev is θ-orthogonal to the tangent (e.g. a
# pure-λ seed meeting a fold head-on) or at a branch point where null(J) is
# 2-dimensional; in those cases — detected by a failed/non-finite LU solve, never
# exercised on the happy path — fall back to the SVD null space, which stays robust
# there. Oriented to continue the previous direction `τprev` (θ-dot ≥ 0).
# `B` and `t` are caller-preallocated scratch reused across every predictor step: `B`
# receives [J; wᵀ] and is factorized IN PLACE (`lu!`), `t` receives e_{n+1} and is
# overwritten by the solution, normalized and oriented in place. The only remaining
# per-step heap allocation on the happy path is `lu!`'s internal pivot vector (O(n)
# Ints); the SVD fallback allocates freely but is never reached on the happy path.
# `J` itself (the Jacobian cache's buffer) is left untouched — `B` holds the copy that
# the factorization destroys — so the fallback can still read it.
function _bordered_tangent!(btc::BorderedTangentCache, J, τprev, wu, wλ, n)
    B, t = btc.B, btc.t
    copyto!(view(B, 1:n, :), J)
    @views B[n + 1, 1:n] .= wu .* τprev[1:n]
    B[n + 1, n + 1] = wλ * τprev[n + 1]
    # The bordered matrix is singular by design at the failure cases (τprev θ-orthogonal
    # to the tangent, branch points), so a singular factorization here is an expected
    # algorithmic signal — caught and routed to the SVD fallback below, unlike the
    # descent methods where it is a genuine error.
    res = try
        btc.lincache(; A = B, b = btc.rhs, linu = t, reuse_A_if_factorization = false)
    catch err
        err isa LinearAlgebra.SingularException || rethrow()
        nothing
    end
    solved = false
    if res !== nothing && res.success
        res.u === t || copyto!(t, res.u)
        # Residual check before accepting: LinearSolve's default solver is itself a
        # polyalgorithm that falls back to pivoted QR on a singular LU and returns a
        # finite LEAST-SQUARES pseudo-solution with a success retcode, so neither the
        # retcode nor finiteness detects the designed-singular cases. A genuine solve
        # has ‖[J t; ⟨w,t⟩ − 1]‖ at rounding level; the inconsistent singular system's
        # pseudo-solution misses by O(1). Computed from `J` and `τprev` (not `B`, which
        # the factorization may have destroyed).
        if all(isfinite, t)
            LinearAlgebra.mul!(btc.r, J, t)
            cres = _theta_dot(t, τprev, wu, wλ, n) - 1
            resid = sqrt(sum(abs2, btc.r) + abs2(cres))
            nt = _theta_norm(t, wu, wλ, n)
            if isfinite(nt) && nt > 0 &&
                    resid <= sqrt(eps(one(nt))) * (1 + nt)
                t ./= nt
                solved = true
            end
        end
    end
    if !solved
        N = LinearAlgebra.nullspace(Matrix(J))
        copyto!(t, view(N, :, 1))
        t ./= _theta_norm(t, wu, wλ, n)
    end
    _theta_dot(t, τprev, wu, wλ, n) < 0 && (t .= .-t)
    return t
end

# Evaluates the Jacobian through the cache (reusing its `J` buffer), solves the bordered
# system into the `t` scratch, and copies the result into `τ` — `τprev === τ` is safe
# because `τ` is only read (border row, orientation sign) before the final copy.
function _arclength_tangent!(btc::BorderedTangentCache, τ, jac_cache, x, wu, wλ, n)
    _bordered_tangent!(btc, jac_cache(x), τ, wu, wλ, n)
    return copyto!(τ, btc.t)
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
        return build_solution_less_specialize(
            prob, alg, copy(prob.u0), start_sol.resid;
            retcode = start_sol.retcode, original = start_sol,
            store_original = alg.store_original
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
    # Packed current point `[u; λ]` and its predecessor: STABLE buffers written in
    # place for the rest of the solve. `xcur` is aliased by the corrector residual
    # built once below, so accepting a step copies through these buffers instead of
    # rebinding them.
    xcur = Vector{Tx}(undef, n + 1)
    copyto!(view(xcur, 1:n), u)
    xcur[n + 1] = λ
    xprev = copy(xcur)                 # no history yet → secant falls back to pure-λ
    have_prev = false

    min_ds = alg.min_ds === nothing ? sqrt(eps(λT)) : λT(alg.min_ds)
    max_ds = λT(alg.max_step_factor) * abs(span)
    ds = min(λT(alg.initial_step_factor) * abs(span), max_ds)
    sλ = λT(sign(span))
    cos_reject = Tx(cos(alg.max_angle))         # turn beyond max_angle ⇒ reject + shrink
    cos_grow = Tx(cos(alg.max_angle / 3))       # turn below max_angle/3 ⇒ allow growth
    streak = 0
    # The tracking cap applies to the augmented corrector solves only; the λ-fixed
    # anchor above and the final landing (`_arclength_fixed_solve`) keep the user's
    # full budget. See `_tracking_budget` (homotopy_sweep.jl) for the resolution rules.
    budget, cap_active = _tracking_budget(alg.tracking_maxiters, prob.kwargs, kwargs)

    # θ-metric weights: every dot/norm below is ⟨⋅,⋅⟩_θ = wu⟨u,u'⟩ + wλ λλ'.
    θw = Tx(alg.theta)
    wu = θw / n
    wλ = one(Tx) - θw
    # pure-λ direction toward λend, unit in the θ metric (‖[0; a]‖_θ = √wλ·|a|).
    τseed = zeros(Tx, n + 1)
    τseed[n + 1] = sλ / sqrt(wλ)

    use_tangent = alg.predictor === :tangent
    # Analytic path Jacobian assembled from the user's jac (nothing without one),
    # shared by the tangent predictor's Jacobian cache and the corrector's Jacobian.
    path_jac = _arclength_path_jac(prob, alg, xcur, n)
    # one reusable Jacobian cache (via NonlinearSolve's own tooling) for the tangent
    # predictor; nothing for the derivative-free secant.
    stats = NLStats(0, 0, 0, 0, 0)
    jac_cache = use_tangent ? _arclength_jac_cache(prob, alg, path_jac, xcur, n, stats) :
        nothing
    # Reused by every predictor step: the bordered-tangent workspace (matrix, rhs,
    # solution, and the shared linear-solver cache) for the tangent, and `tscratch` for
    # the secant chord. Allocated once here so the loop only reuses memory.
    btc = use_tangent ? _bordered_tangent_cache(alg, prob.p, Tx, n, stats) : nothing
    tscratch = Vector{Tx}(undef, n + 1)
    # orientation reference for the tangent predictor: seed toward λend so the first
    # tangent continues into the span (pure-λ direction picks the correct sign). A
    # stable buffer, written in place by every predictor branch below.
    τ = copy(τseed)
    # Derivative fields of the per-step corrector function, built once: the Jacobian
    # aliases the τ buffer (kept in sync with the residual by construction) and the
    # bordered prototypes depend only on the user prototype and n.
    aug_jac = path_jac === nothing ? nothing :
        AugmentedHomotopyJac(path_jac, τ, n, wu, wλ)
    aug_proto = prob.f.jac_prototype === nothing ? nothing :
        _bordered_prototype(prob.f.jac_prototype, n)
    # preserve the constructor-established `sparsity === jac_prototype` identity (see
    # `_arclength_path_function`)
    aug_sparsity = if prob.f.sparsity === nothing
        nothing
    elseif prob.f.sparsity === prob.f.jac_prototype
        aug_proto
    elseif prob.f.sparsity isa AbstractMatrix
        _bordered_prototype(prob.f.sparsity, n)
    else
        prob.f.sparsity
    end

    # ONE corrector residual/function/inner-solver cache, mirroring the sweep's cache
    # driver: the residual reads the in-place-updated `τ`/`xcur` buffers and its `ds`
    # is a mutable field, so each corrector attempt is a couple of buffer/field writes
    # plus a `reinit!` with the new prediction — the inner solver's workspace (Newton
    # state, Jacobian cache, linear-solve cache) is reused across every continuation
    # step instead of being reconstructed. The residual is length n+1 and never
    # in-place even for an iip homotopy — the constraint row has no user-facing
    # buffer, so we always own it. `xpred` is handed to the cache and may be iterated
    # in place when aliasing is forwarded; it is fully rewritten before every attempt.
    # When the tracking cap is active it is baked into this cache — every corrector
    # solve runs capped; the λ-fixed anchor above and the landing solves below are
    # separate full-budget solves — and an explicit user `maxiters` always wins
    # (`cap_active` is false in that case).
    aug = AugmentedHomotopy(prob.f, τ, xcur, Tx(ds), n, wu, wλ)
    augf = _arclength_augmented_function(
        Val(iip), prob.f, aug, aug_jac, aug_proto, aug_sparsity
    )
    xpred = copy(xcur)
    corr_cache = if cap_active
        init(
            NonlinearProblem{iip}(augf, xpred, prob.p), alg.inner, args...;
            prob.kwargs..., kwargs..., maxiters = budget
        )
    else
        init(
            NonlinearProblem{iip}(augf, xpred, prob.p), alg.inner, args...;
            prob.kwargs..., kwargs...
        )
    end
    # Accepted-state buffers reused every step: `ubuf` receives the accepted u-block
    # (the failure returns hand it out), `uland` the interpolated warm start of the
    # λ-fixed landing solve.
    ubuf = Vector{Tx}(undef, n)
    uland = Vector{Tx}(undef, n)

    for _ in 1:(alg.maxsteps)
        # Predictor direction τ (θ-metric unit, length n+1).
        if use_tangent
            # True path tangent (null vector of the augmented Jacobian), oriented to
            # continue τ; accurate from the first step and well-defined at a fold.
            _arclength_tangent!(btc, τ, jac_cache, xcur, wu, wλ, n)
        elseif have_prev
            # Secant through the last two accepted points (chord built in scratch,
            # normalized into the reused τ buffer).
            @. tscratch = xcur - xprev
            dnorm = _theta_norm(tscratch, wu, wλ, n)
            if dnorm > 0
                @. τ = tscratch / dnorm
            else
                copyto!(τ, τseed)
            end
        else
            # Bootstrap: a pure-λ step toward λend (no history for a secant yet).
            copyto!(τ, τseed)
        end

        @. xpred = xcur + ds * τ
        aug.ds = Tx(ds)
        # Retaining reinit!: a polyalgorithm inner resumes from the subalgorithm that
        # won the previous corrector solve (the first corrector runs the full ladder
        # and discovers the winner) instead of re-failing the cheaper ladder members
        # on every warm-started step.
        reinit_retaining!(corr_cache, xpred)
        last_sol = CommonSolve.solve!(corr_cache)

        if SciMLBase.successful_retcode(last_sol)
            xnew = last_sol.u
            # realized chord, built in the (currently free) secant scratch
            @. tscratch = xnew - xcur
            nchord = _theta_norm(tscratch, wu, wλ, n)

            # Curvature control: the realized step direction vs. the predictor measures the
            # path's turn. A large turn means either real high curvature or that the
            # corrector jumped to another branch — both call for a smaller step, so reject
            # and bisect. The gate needs a trustworthy predictor: the tangent is accurate
            # from the first step, but the secant only becomes meaningful once there is
            # history (its pure-λ bootstrap is legitimately misaligned with a sloped branch).
            trust = use_tangent || have_prev
            cosang = (trust && nchord > 0) ?
                clamp(_theta_dot(τ, tscratch, wu, wλ, n) / nchord, -one(Tx), one(Tx)) :
                one(Tx)
            if trust && cosang < cos_reject && alg.adaptive && ds / 2 >= min_ds
                ds = ds / 2
                streak = 0
                continue
            end

            λnew = xnew[n + 1]
            λold = λ

            # Accept: shift the packed history through the stable buffers (the
            # corrector residual aliases `xcur`, so the buffers are copied through,
            # never swapped or rebound) and keep the accepted u-block in `ubuf` for
            # the failure returns.
            copyto!(xprev, xcur)
            copyto!(xcur, xnew)
            copyto!(ubuf, view(xcur, 1:n))
            u = ubuf
            λ = λnew
            have_prev = true

            # A step that brackets λend has crossed the target; land on it exactly with a
            # λ-fixed correction warm-started by interpolation along the just-taken step.
            if (λold - λend) * (λnew - λend) <= 0
                denom = λnew - λold
                frac = denom == 0 ? one(Tx) : Tx((λend - λold) / denom)
                @views @. uland = xprev[1:n] + frac * (xcur[1:n] - xprev[1:n])
                final_sol = _arclength_fixed_solve(
                    prob, alg.inner, uland, λend, args...; kwargs...
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
            # turn it would then have to reject. The growth factor is additionally scaled
            # by the corrector's iteration count (the same AUTO ADPTDS effort bands as the
            # sweep — see `_effort_growth_factor`), giving the arclength driver an effort
            # signal alongside the purely geometric bend-angle test.
            if alg.adaptive
                nit = last_sol.stats === nothing ? -1 : Int(last_sol.stats.nsteps)
                if _effort_wants_shrink(nit, budget)
                    ds / 2 >= min_ds && (ds = ds / 2)
                    streak = 0
                else
                    streak += 1
                    if streak >= alg.expand_threshold && cosang >= cos_grow
                        g = _effort_growth_factor(nit, budget, λT(alg.expand_factor))
                        if g > 1
                            ds = min(g * ds, max_ds)
                            streak = 0
                        end
                    end
                end
            end
        elseif alg.adaptive && ds / 2 >= min_ds
            ds = ds / 2          # bisect; retry from the same accepted point
            streak = 0
        else
            return build_solution_less_specialize(
                prob, alg, u, nothing;
                retcode = last_sol.retcode, original = last_sol,
                store_original = alg.store_original
            )
        end
    end

    # Ran out of attempts without bracketing λend: the path never reached the target.
    return SciMLBase.build_solution(prob, alg, u, nothing; retcode = ReturnCode.MaxIters)
end
