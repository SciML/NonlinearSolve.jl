using NonlinearSolve

using SciMLBase

# --- construction + defaults ---
alg = HomotopyPolyAlgorithm()
@test alg.algs isa Tuple
@test length(alg.algs) == 2
@test alg.algs[1] isa HomotopySweep
@test alg.algs[2] isa ArcLengthContinuation

custom = HomotopyPolyAlgorithm(
    (
        ArcLengthContinuation(; predictor = :tangent),
        HomotopySweep(; predictor = :constant),
    )
)
@test custom.algs[1] isa ArcLengthContinuation
@test custom.algs[1].predictor === :tangent
@test custom.algs[2] isa HomotopySweep
@test custom.algs[2].predictor === :constant

# --- happy path: stage 1 succeeds, stage 2 never runs ---
# Counting residual evaluations proves the arclength stage never ran: the polyalg's
# evaluation count equals a plain HomotopySweep() solve's count exactly (same stage-1
# work, zero stage-2 work).
count_ref = Ref(0)
Hcount = function (u, p, λ)
    count_ref[] += 1
    return [(1 - λ) * (u[1] - 4.0) + λ * (u[1]^2 - 4.0)]
end
prob_happy = HomotopyProblem(Hcount, [4.0]; λspan = (0.0, 1.0))

count_ref[] = 0
sol_poly = solve(prob_happy, HomotopyPolyAlgorithm())
n_poly = count_ref[]
@test SciMLBase.successful_retcode(sol_poly)
@test sol_poly.u[1] ≈ 2.0 atol = 1.0e-6

count_ref[] = 0
sol_sweep = solve(prob_happy, HomotopySweep())
n_sweep = count_ref[]
@test SciMLBase.successful_retcode(sol_sweep)
@test sol_poly.u[1] ≈ sol_sweep.u[1]
@test n_poly == n_sweep

# --- fallback path: stage 1 fails at the fold, stage 2 rounds it ---
# S-shaped fold: u^3 - 3u = -3 + 6λ has turning points at λ = 5/6 and λ = 1/6, so the
# connected branch from the λ = 0 root (u ≈ -2.1038) to the λ = 1 root (u ≈ +2.1038) is
# non-monotone in λ. The natural-parameter sweep cannot reverse λ; the pseudo-arclength
# stage tracks the augmented curve around both folds. With unbounded Newton iterations
# the sweep PATH-JUMPS across the fold onto the upper sheet and reports success (Newton
# on a cubic reaches the far root eventually), so `maxiters = 10` caps the corrector to
# a budget that plenty suffices for on-path warm-started steps but not for the wild
# excursion of a jump — making the sweep fail genuinely (observed: MaxIters at
# u ≈ -1.13, next to the fold at u = -1) while arclength still succeeds.
target = 2.1038034
Hfold(u, p, λ) = [u[1]^3 - 3 * u[1] - (-3 + 6λ)]
prob_fold = HomotopyProblem(Hfold, [-target]; λspan = (0.0, 1.0))

stage1 = HomotopySweep(; inner = SimpleNewtonRaphson(), min_dλ = 1.0e-2)
stage2 = ArcLengthContinuation(; inner = SimpleNewtonRaphson())
sol_stage1 = solve(prob_fold, stage1; maxiters = 10)
@test !SciMLBase.successful_retcode(sol_stage1)     # sweep genuinely fails at the fold

sol_fb = solve(prob_fold, HomotopyPolyAlgorithm((stage1, stage2)); maxiters = 10)
@test SciMLBase.successful_retcode(sol_fb)
@test sol_fb.u[1] ≈ target atol = 1.0e-4            # connected upper-sheet root
@test abs(sol_fb.u[1]^3 - 3 * sol_fb.u[1] - 3) < 1.0e-6

# --- both stages fail: last stage's failed solution is returned ---
# (1-λ)u + λ(u² + 1) has NO real root at λ = 1 (u² + 1 > 0), so no continuation method
# can succeed; the polyalg must report the last stage's failure, with a finite iterate.
Hnone(u, p, λ) = [(1 - λ) * u[1] + λ * (u[1]^2 + 1)]
prob_none = HomotopyProblem(Hnone, [0.0]; λspan = (0.0, 1.0))
sol_none = solve(
    prob_none,
    HomotopyPolyAlgorithm(
        (
            HomotopySweep(; inner = SimpleNewtonRaphson(), min_dλ = 1.0e-2),
            ArcLengthContinuation(; inner = SimpleNewtonRaphson(), maxsteps = 500),
        )
    )
)
@test !SciMLBase.successful_retcode(sol_none)
@test all(isfinite, sol_none.u)
@test sol_none.alg isa ArcLengthContinuation        # the LAST stage's solution

# --- zero-width λspan ---
Hz(u, p, λ) = [u[1]^2 - 4.0]
prob_zero = HomotopyProblem(Hz, [1.5]; λspan = (0.7, 0.7))
sol_zero = solve(prob_zero, HomotopyPolyAlgorithm())
@test SciMLBase.successful_retcode(sol_zero)
@test sol_zero.u[1] ≈ 2.0 atol = 1.0e-8

# --- Float32 eltype is preserved ---
H32(u, p, λ) = [(1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1])]
prob_32 = HomotopyProblem(H32, [4.0f0], [4.0f0]; λspan = (0.0f0, 1.0f0))
sol_32 = solve(prob_32, HomotopyPolyAlgorithm())
@test SciMLBase.successful_retcode(sol_32)
@test eltype(sol_32.u) === Float32
@test sol_32.u[1] ≈ 2.0f0 atol = 1.0e-3
