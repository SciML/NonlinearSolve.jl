"""
    HomotopyPolyAlgorithm(algs::Tuple)
    HomotopyPolyAlgorithm()

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

### Arguments

  - `algs`: a tuple of continuation algorithms to try in order. Each stage must support
    `solve(prob::SciMLBase.HomotopyProblem, alg, args...; kwargs...)`.

### Example

```julia
using NonlinearSolve

alg = HomotopyPolyAlgorithm() # HomotopySweep, then ArcLengthContinuation on failure
alg = HomotopyPolyAlgorithm((
    HomotopySweep(; inner = NewtonRaphson()),
    ArcLengthContinuation(; predictor = :tangent),
))
```
"""
@concrete struct HomotopyPolyAlgorithm <: AbstractNonlinearSolveAlgorithm
    algs <: Tuple
end

function HomotopyPolyAlgorithm()
    return HomotopyPolyAlgorithm((HomotopySweep(), ArcLengthContinuation()))
end

function CommonSolve.solve(
        prob::SciMLBase.HomotopyProblem, alg::HomotopyPolyAlgorithm, args...; kwargs...
    )
    isempty(alg.algs) &&
        throw(ArgumentError("HomotopyPolyAlgorithm requires at least one algorithm"))
    nstages = length(alg.algs)
    for (i, stage) in enumerate(alg.algs)
        sol = CommonSolve.solve(prob, stage, args...; kwargs...)
        (SciMLBase.successful_retcode(sol) || i == nstages) && return sol
    end
    return
end
