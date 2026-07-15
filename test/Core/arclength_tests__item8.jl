using NonlinearSolve

using SciMLBase

# Per-step allocation regression for the corrector cache driver: the inner Newton
# cache, its Jacobian cache, and the linear-solve cache are built ONCE and reused via
# `reinit!`/`solve!` every continuation step, so post-warmup per-step allocation is
# bounded by the inner solver's own iteration internals — rebuilding the corrector
# problem/cache per step (the pre-cache behavior) costs several hundred KB per step at
# n = 50 and blows well past these bounds.
#
# The measurement takes the SLOPE between two fixed-step runs of different lengths:
# compile-time and one-time init allocations cancel, leaving pure per-step cost. The
# in-place homotopy keeps the user residual allocation-free so the bound tracks the
# driver, and `adaptive = false` with a tiny fixed ds makes the step count exactly
# `maxsteps` (the run never brackets λend).
nbig = 50
function Hbig!(r, u, p, λ)
    n = length(u)
    for i in 1:n
        acc = u[i]
        i > 1 && (acc += 0.25 * u[i - 1])
        i < n && (acc += 0.25 * u[i + 1])
        r[i] = acc + λ * u[i]^3 - p.c[i]
    end
    return nothing
end
cbig = [1.0 + 0.25 * ((i > 1) + (i < nbig)) + 1.0 for i in 1:nbig]
prob_big = HomotopyProblem(Hbig!, ones(nbig), (c = cbig,); λspan = (0.0, 1.0))

function run_fixed(prob, predictor, N; inner = nothing)
    alg = ArcLengthContinuation(;
        inner, predictor, adaptive = false, initial_step_factor = 1.0e-4, maxsteps = N
    )
    return solve(prob, alg)
end

function per_step_bytes(prob, predictor; inner = nothing, N1 = 50, N2 = 250)
    sol = run_fixed(prob, predictor, 5; inner)   # warm up compilation
    @test sol.retcode == SciMLBase.ReturnCode.MaxIters
    run_fixed(prob, predictor, 5; inner)
    GC.gc()
    a1 = @allocated run_fixed(prob, predictor, N1; inner)
    GC.gc()
    a2 = @allocated run_fixed(prob, predictor, N2; inner)
    return (a2 - a1) / (N2 - N1)
end

# With the corrector cache (built once, re-driven via `reinit!`/`solve!`) plus the Newton
# stack's own allocation-free reuse (#1038 dense-LU alias, #1039 `linsolve_workspace`)
# landed, the DEFAULT polyalgorithm inner measures ≈ 2.5 KB/step (secant) / 3.0 KB/step
# (tangent) at n = 50 on Julia 1.10 (lower on 1.11, where LinearSolve's ipiv reuse is
# already active). These bounds sit at roughly 2× that, well under the pre-cache driver's
# ≈ 274/295 KB, so they guard the cache-reuse invariant while tolerating CI variance.
@test per_step_bytes(prob_big, :secant) < 6_000
@test per_step_bytes(prob_big, :tangent) < 7_000

# Driver-isolating guard with a single-method NewtonRaphson inner (one LU solve/step, no
# polyalgorithm ladder), so the per-step number is dominated by the driver glue and one
# linear solve. This is what the `last_sol` union-boxing fix (reading the interior
# corrector solution through `@noinline` typed field barriers so the union-typed
# `last_sol.u`/`.stats` reads do not box) directly bounds: on master the boxing added
# ~96 B/step here. Measured ≈ 720 B/step (secant) / 1250 B/step (tangent) on Julia 1.10
# (≈ 224/256 B/step on 1.11); the ~500 B/step 1.10 gap is LinearSolve's Julia-1.10-only
# refactorization ipiv (a fresh pivot vector per `lu!`, reused only on Julia ≥ 1.11 or
# with the LinearSolve backport). Bounds catch a return of the union boxing while holding
# across both Julia versions.
newton_inner = NewtonRaphson()
@test per_step_bytes(prob_big, :secant; inner = newton_inner) < 1_100
@test per_step_bytes(prob_big, :tangent; inner = newton_inner) < 1_700

# the fixed-step measurement configuration must not change the answer: the standard
# adaptive solve still lands on u = ones(n)
for predictor in (:secant, :tangent)
    sol = solve(prob_big, ArcLengthContinuation(; predictor))
    @test SciMLBase.successful_retcode(sol)
    @test maximum(abs, sol.u .- 1.0) < 1.0e-6
end
