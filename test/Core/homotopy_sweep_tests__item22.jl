using NonlinearSolve

using SciMLBase

# Per-step allocation regression for the HomotopySweep driver glue: the inner-solver
# cache is built ONCE and re-driven every continuation step (`reinit!`/`solve!`), so
# post-warmup per-step allocation is bounded by the inner solver's own iteration
# internals plus the sweep's accept/quality glue. The accept/quality block reads the
# converged iterate through a single `solu = last_sol.u` local instead of four bare
# `last_sol.u` accesses; `last_sol` is union-typed across the anchor/step branches, so
# each bare access boxed a fresh getproperty — the local drops three redundant boxed
# getproperty calls per step (≈1.5 KB/step at n = 50).
#
# The measurement takes the SLOPE between two fixed-step runs of different lengths:
# compile-time and one-time init allocations cancel, leaving pure per-step cost. The
# in-place homotopy keeps the user residual allocation-free so the bound tracks the
# driver, and `adaptive = false` makes the step count exactly `nsteps`.
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

function run_fixed(prob, N)
    return solve(prob, HomotopySweep(; adaptive = false, nsteps = N))
end

function per_step_bytes(prob; N1 = 50, N2 = 250)
    sol = run_fixed(prob, 5)   # warm up compilation
    @test SciMLBase.successful_retcode(sol)
    run_fixed(prob, 5)
    GC.gc()
    a1 = @allocated run_fixed(prob, N1)
    GC.gc()
    a2 = @allocated run_fixed(prob, N2)
    return (a2 - a1) / (N2 - N1)
end

# Measured ≈ 6.0 KB/step at n = 50 after the single-getproperty accept/quality glue;
# the pre-change driver measured ≈ 7.5 KB/step (the extra ≈1.5 KB being the three
# redundant boxed `last_sol.u` getproperty calls per step). The remainder is the shared
# Newton stack (LinearSolve refactorization copy, Broyden/Klement internals of the
# default inner polyalgorithm) plus the one per-step SciMLBase getproperty (≈0.4 KB)
# and LinearSolve solution-wrapper (≈0.1 KB) allocations that separate efforts remove.
# The bound sits between the two, so it catches a regression that reintroduces the
# redundant getproperty accesses while leaving headroom for inner-stack variance.
@test per_step_bytes(prob_big) < 6800

# the fixed-step measurement configuration must not change the answer: the standard
# adaptive solve still lands on u = ones(n)
sol = solve(prob_big, HomotopySweep())
@test SciMLBase.successful_retcode(sol)
@test maximum(abs, sol.u .- 1.0) < 1.0e-6
