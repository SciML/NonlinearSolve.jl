using NonlinearSolve

using SciMLBase

# Per-step allocation regression for the HomotopySweep driver, mirroring
# `arclength_tests__item8.jl`. The inner solver's cache (Newton cache, Jacobian cache,
# linear-solve cache) is built ONCE and re-driven via `reinit!`/`solve!` every
# continuation step, so post-warmup per-step allocation is bounded by the inner solver's
# own internals plus the driver's own glue — reconstructing the inner problem/cache per
# step (the pre-cache behavior) costs several hundred KB per step at n = 50.
#
# In addition, this guards the `last_sol` union-boxing fix: the interior accept/quality
# block reads the converged corrector iterate through `@noinline` typed field barriers
# (`_sweep_sol_u`/`_sweep_sol_nsteps`) instead of bare `last_sol.u`/`last_sol.stats`
# reads. `last_sol` is union-typed across the concrete `solve!(cache)` and the fresh
# `solve(inner_prob, inner)` exempt anchor/landing solves, so the bare reads boxed a fresh
# getproperty on every step (four reads/step in the sweep, ~384 B/step at n = 50). The
# barriers force the union-split at the call so the reads no longer box.
#
# The measurement takes the SLOPE between two fixed-step runs of different lengths so
# compile-time and one-time init allocations cancel, leaving pure per-step cost.
# `adaptive = false` with an explicit `nsteps` makes the step count exactly `nsteps`.
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

function run_fixed(prob, N; inner = nothing)
    alg = HomotopySweep(; inner, adaptive = false, nsteps = N)
    return solve(prob, alg)
end

function per_step_bytes(prob; inner = nothing, N1 = 50, N2 = 250)
    run_fixed(prob, 5; inner)   # warm up compilation
    run_fixed(prob, 5; inner)
    GC.gc()
    a1 = @allocated run_fixed(prob, N1; inner)
    GC.gc()
    a2 = @allocated run_fixed(prob, N2; inner)
    return (a2 - a1) / (N2 - N1)
end

# DEFAULT polyalgorithm inner: ≈ 2.9 KB/step at n = 50 on Julia 1.10 (lower on 1.11).
# The bound sits at roughly 2× that, well under the pre-cache several-hundred-KB level.
@test per_step_bytes(prob_big) < 7_000

# Driver-isolating guard with a single-method NewtonRaphson inner (one LU solve/step, no
# polyalgorithm ladder). This is what the `last_sol` union-boxing fix directly bounds:
# on master the four boxed `last_sol.u` reads added ~384 B/step here. Measured ≈ 1245
# B/step on Julia 1.10 (≈ 96 B/step on 1.11); the ~1150 B/step 1.10 gap is LinearSolve's
# Julia-1.10-only refactorization ipiv (reused on 1.11 / with the LinearSolve backport).
# The bound catches a return of the union boxing (which would push this back to ~1629
# B/step on 1.10) while holding across both Julia versions.
@test per_step_bytes(prob_big; inner = NewtonRaphson()) < 1_500

# the fixed-step measurement configuration must not change the answer: the standard
# adaptive solve still lands on u = ones(n)
sol = solve(prob_big, HomotopySweep())
@test SciMLBase.successful_retcode(sol)
@test maximum(abs, sol.u .- 1.0) < 1.0e-6
