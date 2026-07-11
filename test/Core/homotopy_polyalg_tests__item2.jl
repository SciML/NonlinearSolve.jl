using NonlinearSolve

using NonlinearSolveBase
using SciMLBase

# --- warm_handoff construction ---
@test HomotopyPolyAlgorithm().warm_handoff
@test !HomotopyPolyAlgorithm(; warm_handoff = false).warm_handoff
@test HomotopyPolyAlgorithm((HomotopySweep(),); warm_handoff = false).warm_handoff == false

# S-shaped fold: u^3 - 3u = -3 + 6λ has turning points at λ = 5/6 (u = -1) and λ = 1/6
# (u = +1), so the connected branch from the λ = 0 root (u ≈ -2.1038) to the λ = 1 root
# (u ≈ +2.1038) is non-monotone in λ. The sweep fails at the λ = 5/6 fold (with
# `maxiters = 10` capping the corrector below what a path-jump excursion needs); the
# arclength fallback rounds it. This is the warm-handoff scenario: the sweep conquers
# λ ∈ [0, ~0.83] before dying, and the fallback should not redo that prefix.
target = 2.1038034
count_ref = Ref(0)
Hfold = function (u, p, λ)
    count_ref[] += 1
    return [u[1]^3 - 3 * u[1] - (-3 + 6λ)]
end
prob_fold = SciMLBase.HomotopyProblem(Hfold, [-target]; λspan = (0.0, 1.0))
stage1 = HomotopySweep(; inner = SimpleNewtonRaphson(), min_dλ = 1.0e-2)
stage2 = ArcLengthContinuation(; inner = SimpleNewtonRaphson())

# --- warm handoff: succeeds, correct root, fewer residual calls than cold restart ---
count_ref[] = 0
sol_warm = solve(prob_fold, HomotopyPolyAlgorithm((stage1, stage2)); maxiters = 10)
n_warm = count_ref[]
@test SciMLBase.successful_retcode(sol_warm)
@test sol_warm.u[1] ≈ target atol = 1.0e-4

count_ref[] = 0
sol_cold = solve(
    prob_fold, HomotopyPolyAlgorithm((stage1, stage2); warm_handoff = false);
    maxiters = 10
)
n_cold = count_ref[]
@test SciMLBase.successful_retcode(sol_cold)
@test sol_cold.u[1] ≈ target atol = 1.0e-4

# the headline claim: the warm-started fallback skips the sweep-conquered prefix and
# costs strictly fewer residual evaluations than restarting the fallback cold
@test n_warm < n_cold

# --- the warm success corresponds to the ORIGINAL problem ---
@test sol_warm.prob === prob_fold
@test typeof(sol_warm.u) == typeof(prob_fold.u0)
# by default `original` is dropped so the returned solution stays concretely typed
@test fieldtype(typeof(sol_warm), :original) === Nothing

# with `store_original = Val(true)` the stage's solution of the shrunken
# remaining-stretch problem is kept as `original`: its span starts strictly inside the
# original span and ends at the original target, backed off ~5% of the span from the
# sweep's last accepted λ (which is near the λ = 5/6 fold)
sol_warm_orig = solve(
    prob_fold, HomotopyPolyAlgorithm((stage1, stage2); store_original = Val(true));
    maxiters = 10
)
@test SciMLBase.successful_retcode(sol_warm_orig)
@test sol_warm_orig.u[1] ≈ target atol = 1.0e-4
hsol = sol_warm_orig.original
@test hsol.prob isa SciMLBase.HomotopyProblem
@test hsol.prob.λspan[2] == 1.0
@test 0.0 < hsol.prob.λspan[1] < 5 / 6
@test hsol.u == sol_warm_orig.u

# --- anchor failure: no accepted point, warm handoff must not engage ---
# (1 - λ)(atan(u) + 2) + λ(u - 1) has no real root at λ = 0 (atan(u) + 2 ∈ (0.43, 3.57))
# and a never-singular Jacobian, so the sweep's anchor solve fails cleanly and there is
# nothing to hand off; with and without warm_handoff the polyalg must do exactly the
# same (cold) work.
count_ref[] = 0
Hanchor = function (u, p, λ)
    count_ref[] += 1
    return [(1 - λ) * (atan(u[1]) + 2) + λ * (u[1] - 1)]
end
prob_anchor = SciMLBase.HomotopyProblem(Hanchor, [0.0]; λspan = (0.0, 1.0))
count_ref[] = 0
sol_aw = solve(prob_anchor, HomotopyPolyAlgorithm((stage1, stage2)); maxiters = 10)
n_aw = count_ref[]
count_ref[] = 0
sol_ac = solve(
    prob_anchor, HomotopyPolyAlgorithm((stage1, stage2); warm_handoff = false);
    maxiters = 10
)
n_ac = count_ref[]
@test !SciMLBase.successful_retcode(sol_aw)
@test !SciMLBase.successful_retcode(sol_ac)
@test n_aw == n_ac

# --- progress below the backoff width: handoff must not engage ---
# Anchoring the same fold problem at λ0 = 0.83 puts the λ = 5/6 fold within 2% of the
# span from the anchor, closer than the 5% backoff, so the backed-off λ_h would land
# before λspan[1]. First confirm the driver scenario (an interior accepted point DOES
# exist — this is not the anchor-failure case), then that warm and cold do identical
# work. u0 ≈ the lower-branch root at λ = 0.83.
prob_close = SciMLBase.HomotopyProblem(Hfold, [-1.0801]; λspan = (0.83, 1.0))
stage1_fine = HomotopySweep(; inner = SimpleNewtonRaphson(), min_dλ = 1.0e-3)
csol, λ_last = NonlinearSolveBase._homotopy_sweep_solve(
    prob_close, stage1_fine; maxiters = 10
)
@test !SciMLBase.successful_retcode(csol)
@test λ_last !== nothing
@test 0.83 < λ_last < 0.83 + 0.05 * (1.0 - 0.83)   # interior, but within the backoff
count_ref[] = 0
sol_cw = solve(prob_close, HomotopyPolyAlgorithm((stage1_fine, stage2)); maxiters = 10)
n_cw = count_ref[]
count_ref[] = 0
sol_cc = solve(
    prob_close, HomotopyPolyAlgorithm((stage1_fine, stage2); warm_handoff = false);
    maxiters = 10
)
n_cc = count_ref[]
@test n_cw == n_cc
@test SciMLBase.successful_retcode(sol_cw) == SciMLBase.successful_retcode(sol_cc)

# --- double fallback: warm attempt fails, cold full-range attempt still runs ---
# A stage that fails on any shrunken span but succeeds on the original one: the warm
# attempt (seeded near the fold) must not consume the stage's only try — the polyalg
# retries it cold before moving on.
struct PickySpanStage
    spans_seen::Vector{Float64}
end
function SciMLBase.solve(
        prob::SciMLBase.HomotopyProblem, alg::PickySpanStage, args...; kwargs...
    )
    push!(alg.spans_seen, prob.λspan[1])
    if prob.λspan[1] != 0.0
        return SciMLBase.build_solution(
            prob, alg, copy(prob.u0), nothing; retcode = SciMLBase.ReturnCode.Failure
        )
    end
    return SciMLBase.solve(
        prob, ArcLengthContinuation(; inner = SimpleNewtonRaphson()), args...; kwargs...
    )
end
picky = PickySpanStage(Float64[])
sol_dbl = solve(prob_fold, HomotopyPolyAlgorithm((stage1, picky)); maxiters = 10)
@test SciMLBase.successful_retcode(sol_dbl)
@test sol_dbl.u[1] ≈ target atol = 1.0e-4
# the stage was attempted warm first (shrunken span start > 0), then cold (start == 0)
@test length(picky.spans_seen) == 2
@test picky.spans_seen[1] > 0.0
@test picky.spans_seen[2] == 0.0

# --- Float32 warm handoff preserves the eltype ---
Hfold32 = (u, p, λ) -> [u[1]^3 - 3 * u[1] - (-3 + 6λ)]
prob_32 = SciMLBase.HomotopyProblem(Hfold32, [-Float32(target)]; λspan = (0.0f0, 1.0f0))
sol_32 = solve(
    prob_32,
    HomotopyPolyAlgorithm(
        (
            HomotopySweep(; inner = SimpleNewtonRaphson(), min_dλ = 1.0f-2),
            ArcLengthContinuation(; inner = SimpleNewtonRaphson()),
        )
    );
    maxiters = 10
)
@test SciMLBase.successful_retcode(sol_32)
@test eltype(sol_32.u) === Float32
@test sol_32.u[1] ≈ Float32(target) atol = 1.0f-3
