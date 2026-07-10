using NonlinearSolve

using SciMLBase
using StaticArrays

# `tracking_maxiters` caps the inner corrector's iteration budget on interior tracking
# steps, making rejections cheap: on a fold problem where the sweep must fail
# (u² + λ − 0.5 = 0 has no real solution past λ = 0.5), every bisection retry otherwise
# re-runs the inner solver against its full default budget of 1000 iterations.
# Reference measurements (Julia 1.11, this commit): default polyalg inner burns 93049
# residual calls uncapped versus 3257 at the default cap of 20;
# NewtonRaphson burns 48132 uncapped and 1050 at 20 — matching the ~80× reduction
# measured in SciML/NonlinearSolve.jl#1020. The thresholds below carry generous margin
# over those measurements so benign solver-internal drift does not flip them.
nf = Ref(0)
H(u, p, λ) = (nf[] += 1; [u[1]^2 + λ - 0.5])
prob = HomotopyProblem(H, [0.7]; λspan = (0.0, 1.0))

nf[] = 0
sol = solve(prob, HomotopySweep(; tracking_maxiters = nothing))
@test !SciMLBase.successful_retcode(sol)
nf_uncapped = nf[]

nf[] = 0
sol = solve(prob, HomotopySweep())
@test !SciMLBase.successful_retcode(sol)
nf_default = nf[]

nf[] = 0
sol = solve(prob, HomotopySweep(; tracking_maxiters = 20))
@test !SciMLBase.successful_retcode(sol)
nf_cap20 = nf[]

@test nf_cap20 < nf_uncapped ÷ 10
@test nf_default < nf_uncapped ÷ 4

# an explicit user-passed `maxiters` (solve kwargs) always wins over the tracking cap,
# resurrecting the full-budget behavior
nf[] = 0
sol = solve(prob, HomotopySweep(; tracking_maxiters = 20); maxiters = 1000)
@test !SciMLBase.successful_retcode(sol)
@test nf[] > 5 * nf_cap20

# and a problem-level `maxiters` wins the same way
nf[] = 0
probkw = HomotopyProblem(H, [0.7]; λspan = (0.0, 1.0), maxiters = 1000)
sol = solve(probkw, HomotopySweep(; tracking_maxiters = 20))
@test !SciMLBase.successful_retcode(sol)
@test nf[] > 5 * nf_cap20

# same reduction with a plain Newton inner (the issue's second measurement)
nf[] = 0
sol = solve(prob, HomotopySweep(; inner = NewtonRaphson(), tracking_maxiters = nothing))
@test !SciMLBase.successful_retcode(sol)
nf_nr_uncapped = nf[]
nf[] = 0
sol = solve(prob, HomotopySweep(; inner = NewtonRaphson(), tracking_maxiters = 20))
@test !SciMLBase.successful_retcode(sol)
@test nf[] < nf_nr_uncapped ÷ 10

# SimpleHomotopySweep: same cap semantics on the value-oriented driver
nfs = Ref(0)
Hs(u, p, λ) = (nfs[] += 1; SA[u[1]^2 + λ - 0.5])
probs = HomotopyProblem(Hs, SA[0.7]; λspan = (0.0, 1.0))
nfs[] = 0
sol = solve(probs, SimpleHomotopySweep(; tracking_maxiters = nothing))
@test !SciMLBase.successful_retcode(sol)
nfs_uncapped = nfs[]
nfs[] = 0
sol = solve(probs, SimpleHomotopySweep(; tracking_maxiters = 20))
@test !SciMLBase.successful_retcode(sol)
@test nfs[] < nfs_uncapped ÷ 10

# The cap must NOT apply to the λspan[1] anchor solve: from u0 = 100 the anchor's cold
# Newton run needs ~14 iterations (verified below), far above the cap of 5, yet the
# sweep succeeds because the anchor is exempt. The warm-started interior steps fit
# comfortably inside the cap.
Ha(u, p, λ) = [u[1]^3 - 1 - λ]
proba = HomotopyProblem(Ha, [100.0]; λspan = (0.0, 1.0))
anchor_direct = solve(
    NonlinearProblem((u, p) -> [u[1]^3 - 1], [100.0]), NewtonRaphson(); maxiters = 5
)
@test !SciMLBase.successful_retcode(anchor_direct)
sol = solve(proba, HomotopySweep(; inner = NewtonRaphson(), tracking_maxiters = 5))
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ cbrt(2) atol = 1.0e-8

Has(u, p, λ) = SA[u[1]^3 - 1 - λ]
probas = HomotopyProblem(Has, SA[100.0]; λspan = (0.0, 1.0))
sol = solve(probas, SimpleHomotopySweep(; tracking_maxiters = 5))
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ cbrt(2) atol = 1.0e-8

# The cap must NOT apply to the final step that lands on λspan[2]: with a single fixed
# step (nsteps = 1, adaptive = false) the only step IS the landing, and it needs more
# than the 1 iteration the cap would allow (verified below), yet the sweep succeeds.
# adaptive = false means a capped landing would fail the sweep outright, so success
# proves the exemption.
Hl(u, p, λ) = [u[1]^3 - 1 - 7λ]
probl = HomotopyProblem(Hl, [1.0]; λspan = (0.0, 1.0))
landing_direct = solve(
    NonlinearProblem((u, p) -> [u[1]^3 - 8], [1.0]), NewtonRaphson(); maxiters = 1
)
@test !SciMLBase.successful_retcode(landing_direct)
sol = solve(
    probl,
    HomotopySweep(;
        inner = NewtonRaphson(), nsteps = 1, adaptive = false, tracking_maxiters = 1
    )
)
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 2.0 atol = 1.0e-8

Hls(u, p, λ) = SA[u[1]^3 - 1 - 7λ]
probls = HomotopyProblem(Hls, SA[1.0]; λspan = (0.0, 1.0))
sol = solve(
    probls, SimpleHomotopySweep(; nsteps = 1, adaptive = false, tracking_maxiters = 1)
)
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 2.0 atol = 1.0e-8

# constructor validation
@test_throws ArgumentError HomotopySweep(; tracking_maxiters = 0)
@test_throws ArgumentError SimpleHomotopySweep(; tracking_maxiters = 0)
@test_throws ArgumentError ArcLengthContinuation(; tracking_maxiters = 0)
