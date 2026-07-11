using NonlinearSolve

using SciMLBase
using StaticArrays

# `tracking_abstol` loosens the inner corrector's tolerance on interior tracking steps
# only: interior iterates just need to stay inside the next step's Newton convergence
# basin, while the λspan[1] anchor and the final landing on λspan[2] always run at the
# user's full tolerances, so the returned solution's accuracy semantics are unchanged.
# Reference measurements (Julia 1.12, this commit) on u³ - 1 - λ with a NewtonRaphson
# inner: 53 residual calls tight vs 43 at tracking_abstol = 1e-3 (19%); on the n = 50
# coupled cubic: 1247 vs 835 (33%); SimpleHomotopySweep scalar path: 43 vs 27 (37%).
# The thresholds below carry generous margin over those measurements.

# ---- (a) + (b): final solution stays tight, residual-call count drops ----
nf = Ref(0)
H(u, p, λ) = (nf[] += 1; [u[1]^3 - 1 - λ])
prob = HomotopyProblem(H, [1.0]; λspan = (0.0, 1.0))

nf[] = 0
sol = solve(prob, HomotopySweep(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(sol)
nf_tight = nf[]

nf[] = 0
sol = solve(prob, HomotopySweep(; inner = NewtonRaphson(), tracking_abstol = 1.0e-3))
@test SciMLBase.successful_retcode(sol)
# despite the loose interior tracking, the landing runs at the full tolerance
@test sol.u[1] ≈ cbrt(2) atol = 1.0e-9
nf_loose = nf[]
@test nf_loose < nf_tight

# the default (tracking_abstol = nothing) must not change behavior at all
nf[] = 0
sol_def = solve(prob, HomotopySweep(; inner = NewtonRaphson(), tracking_abstol = nothing))
@test nf[] == nf_tight

# stronger margin on the n = 50 coupled cubic system (measured 33% reduction)
nfn = Ref(0)
function Hn(du, u, p, λ)
    nfn[] += 1
    for i in 1:length(u)
        um = i == 1 ? zero(eltype(u)) : u[i - 1]
        up = i == length(u) ? zero(eltype(u)) : u[i + 1]
        du[i] = (1 - λ) * (u[i] - p) + λ * (3u[i] - um - up + u[i]^3 - p)
    end
    return nothing
end
probn = HomotopyProblem(NonlinearFunction{true}(Hn), fill(1.0, 50), 1.0)
nfn[] = 0
soln = solve(probn, HomotopySweep(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(soln)
nfn_tight = nfn[]
nfn[] = 0
soln = solve(probn, HomotopySweep(; inner = NewtonRaphson(), tracking_abstol = 1.0e-3))
@test SciMLBase.successful_retcode(soln)
nfn_loose = nfn[]
@test nfn_loose < (nfn_tight * 85) ÷ 100
residn = zeros(50)
Hn(residn, soln.u, 1.0, 1.0)
@test maximum(abs, residn) < 1.0e-9   # landing at full tolerance, not 1e-3

# SimpleHomotopySweep: same semantics on the value-oriented driver
nfs = Ref(0)
Hs(u, p, λ) = (nfs[] += 1; SA[u[1]^3 - 1 - λ])
probs = HomotopyProblem(Hs, SA[1.0]; λspan = (0.0, 1.0))
nfs[] = 0
sols = solve(probs, SimpleHomotopySweep())
@test SciMLBase.successful_retcode(sols)
nfs_tight = nfs[]
nfs[] = 0
sols = solve(probs, SimpleHomotopySweep(; tracking_abstol = 1.0e-3))
@test SciMLBase.successful_retcode(sols)
@test sols.u[1] ≈ cbrt(2) atol = 1.0e-9
nfs_loose = nfs[]
@test nfs_loose < (nfs_tight * 85) ÷ 100

# ---- (c) anchor exemption ----
# Zero-width λspan: the anchor IS the returned solve. A standalone solve at the loose
# tolerance stops far from the root (verified directly below), yet the sweep's result
# is tight because the anchor is exempt from tracking_abstol.
Ha(u, p, λ) = [u[1]^3 - 1 - λ]
proba = HomotopyProblem(Ha, [3.0]; λspan = (0.0, 0.0))
loose_direct = solve(
    NonlinearProblem((u, p) -> [u[1]^3 - 1], [3.0]), NewtonRaphson(); abstol = 0.5
)
@test abs(loose_direct.u[1] - 1) > 1.0e-3
for alg in (
        HomotopySweep(; inner = NewtonRaphson(), tracking_abstol = 0.5),
        HomotopySweep(; tracking_abstol = 0.5),
    )
    sola = solve(proba, alg)
    @test SciMLBase.successful_retcode(sola)
    @test sola.u[1] ≈ 1.0 atol = 1.0e-8
end

Has(u, p, λ) = SA[u[1]^3 - 1 - λ]
probas = HomotopyProblem(Has, SA[3.0]; λspan = (0.0, 0.0))
solas = solve(probas, SimpleHomotopySweep(; tracking_abstol = 0.5))
@test SciMLBase.successful_retcode(solas)
@test solas.u[1] ≈ 1.0 atol = 1.0e-8

# ---- (c) landing exemption ----
# With a single fixed step (nsteps = 1, adaptive = false) the only step IS the landing.
# A standalone solve at the loose tolerance stops far from the λ = 1 root u = 2
# (verified directly below), yet the sweep's landing is exempt and returns it tight.
Hl(u, p, λ) = [u[1]^3 - 1 - 7λ]
probl = HomotopyProblem(Hl, [1.0]; λspan = (0.0, 1.0))
loose_landing_direct = solve(
    NonlinearProblem((u, p) -> [u[1]^3 - 8], [1.0]), NewtonRaphson(); abstol = 0.5
)
@test abs(loose_landing_direct.u[1] - 2) > 1.0e-3
sol = solve(
    probl,
    HomotopySweep(;
        inner = NewtonRaphson(), nsteps = 1, adaptive = false, tracking_abstol = 0.5
    )
)
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 2.0 atol = 1.0e-8

Hls(u, p, λ) = SA[u[1]^3 - 1 - 7λ]
probls = HomotopyProblem(Hls, SA[1.0]; λspan = (0.0, 1.0))
sol = solve(
    probls, SimpleHomotopySweep(; nsteps = 1, adaptive = false, tracking_abstol = 0.5)
)
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 2.0 atol = 1.0e-8

# ---- (d) explicit user tolerances win over the tracking default ----
# With a user-passed abstol the loosening is disabled outright, so the run is
# identical (deterministic solver, exact same code path) to tracking_abstol = nothing.
nf[] = 0
solve(prob, HomotopySweep(; inner = NewtonRaphson()); abstol = 1.0e-12)
nf_user = nf[]
nf[] = 0
solve(prob, HomotopySweep(; inner = NewtonRaphson(), tracking_abstol = 1.0e-3); abstol = 1.0e-12)
@test nf[] == nf_user

# problem-level abstol wins the same way
probkw = HomotopyProblem(H, [1.0]; λspan = (0.0, 1.0), abstol = 1.0e-12)
nf[] = 0
solve(probkw, HomotopySweep(; inner = NewtonRaphson()))
nf_probkw = nf[]
nf[] = 0
solve(probkw, HomotopySweep(; inner = NewtonRaphson(), tracking_abstol = 1.0e-3))
@test nf[] == nf_probkw

# a user reltol also disables the loosening (a loose abstol spliced next to a user
# reltol would fire first in the OR-combined default termination modes)
nf[] = 0
solve(prob, HomotopySweep(; inner = NewtonRaphson()); reltol = 1.0e-10)
nf_rel = nf[]
nf[] = 0
solve(prob, HomotopySweep(; inner = NewtonRaphson(), tracking_abstol = 1.0e-3); reltol = 1.0e-10)
@test nf[] == nf_rel

# SimpleHomotopySweep: user abstol wins identically
nfs[] = 0
solve(probs, SimpleHomotopySweep(); abstol = 1.0e-12)
nfs_user = nfs[]
nfs[] = 0
solve(probs, SimpleHomotopySweep(; tracking_abstol = 1.0e-3); abstol = 1.0e-12)
@test nfs[] == nfs_user

# ---- (e) constructor validation ----
@test_throws ArgumentError HomotopySweep(; tracking_abstol = 0.0)
@test_throws ArgumentError HomotopySweep(; tracking_abstol = -1.0e-3)
@test_throws ArgumentError SimpleHomotopySweep(; tracking_abstol = 0.0)
@test_throws ArgumentError SimpleHomotopySweep(; tracking_abstol = -1.0e-3)
@test HomotopySweep(; tracking_abstol = nothing) isa HomotopySweep
@test SimpleHomotopySweep(; tracking_abstol = 1.0e-4) isa SimpleHomotopySweep
