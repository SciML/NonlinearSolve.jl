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

function run_fixed(prob, predictor, N)
    alg = ArcLengthContinuation(;
        predictor, adaptive = false, initial_step_factor = 1.0e-4, maxsteps = N
    )
    return solve(prob, alg)
end

function per_step_bytes(prob, predictor; N1 = 50, N2 = 250)
    sol = run_fixed(prob, predictor, 5)   # warm up compilation
    @test sol.retcode == SciMLBase.ReturnCode.MaxIters
    run_fixed(prob, predictor, 5)
    GC.gc()
    a1 = @allocated run_fixed(prob, predictor, N1)
    GC.gc()
    a2 = @allocated run_fixed(prob, predictor, N2)
    return (a2 - a1) / (N2 - N1)
end

# Measured ≈ 56 KB (secant) / 77 KB (tangent) per step at n = 50 with the cache
# driver; the driver's own glue profiles at < 1 KB/step. The remainder is the shared
# Newton stack, attributed by Profile.Allocs: ~21.5 KB/step is LinearSolve's
# `lu(A, check = false)` copy on refactorization (skippable via its in-place `lu!`
# path, gated on `alias_A = true` at init — a `construct_linear_solver` follow-up,
# not driver work), ~21.5 KB/step was Broyden's initial J⁻¹ (a fresh `lu(A)` per step,
# since replaced by the reusable `Utils.linsolve_workspace` linear-solve
# cache), and ~10 KB/step is Klement/termination internals of the default
# inner polyalgorithm. The pre-cache driver measured ≈ 274 KB / 295 KB. Bounds sit
# at roughly 2× the cached cost, well under half the uncached cost, and hold
# regardless of whether the Newton-stack follow-ups land.
@test per_step_bytes(prob_big, :secant) < 128_000
@test per_step_bytes(prob_big, :tangent) < 160_000

# the fixed-step measurement configuration must not change the answer: the standard
# adaptive solve still lands on u = ones(n)
for predictor in (:secant, :tangent)
    sol = solve(prob_big, ArcLengthContinuation(; predictor))
    @test SciMLBase.successful_retcode(sol)
    @test maximum(abs, sol.u .- 1.0) < 1.0e-6
end
