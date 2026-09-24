using NonlinearSolve

using SciMLBase

# ---- construction + validation of the tangent-predictor options ----
alg = ArcLengthContinuation()
@test alg.predictor === :secant          # default unchanged
@test alg.autodiff === nothing

alg_t = ArcLengthContinuation(; predictor = :tangent, autodiff = AutoForwardDiff())
@test alg_t.predictor === :tangent
@test alg_t.autodiff isa AutoForwardDiff

@test_throws ArgumentError ArcLengthContinuation(; predictor = :bogus)
@test_throws ArgumentError ArcLengthContinuation(; predictor = :constant)

# ---- the tangent predictor rounds the S-fold to the connected upper-sheet root ----
# u^3 - 3u = -3 + 6λ; folds at u = ±1. Connected branch from the λ=0 lower root
# (u ≈ -2.1038) reaches the λ=1 upper root (u ≈ +2.1038) only by rounding both folds.
target = 2.1038034
Hfold(u, p, λ) = [u[1]^3 - 3 * u[1] - (-3 + 6λ)]
prob = HomotopyProblem(Hfold, [-target]; λspan = (0.0, 1.0))

sol_t = solve(prob, ArcLengthContinuation(; predictor = :tangent))
@test SciMLBase.successful_retcode(sol_t)
@test sol_t.u[1] ≈ target atol = 1.0e-4
@test abs(sol_t.u[1]^3 - 3 * sol_t.u[1] - 3) < 1.0e-6   # genuine target-system residual

# tangent and secant predictors land on the same connected-branch root
sol_s = solve(prob, ArcLengthContinuation(; predictor = :secant))
@test sol_t.u[1] ≈ sol_s.u[1] atol = 1.0e-4

# ---- correctness carries over to monotone, multi-dim, and Float32 ----
Hm(u, p, λ) = [(1 - λ) * (u[1] - p.c) + λ * (u[1]^2 - p.c)]
sm = solve(HomotopyProblem(Hm, [4.0], (c = 4.0,)), ArcLengthContinuation(; predictor = :tangent))
@test SciMLBase.successful_retcode(sm)
@test sm.u[1] ≈ 2.0 atol = 1.0e-6

# 2D: u2 = u1^2 slaved to the folding u1 — the tangent must respect the augmented
# (n+1)-dimensional Jacobian null space, not just a scalar slope.
H2(u, p, λ) = [u[1]^3 - 3 * u[1] - (-3 + 6λ), u[2] - u[1]^2]
s2 = solve(
    HomotopyProblem(H2, [-target, target^2]; λspan = (0.0, 1.0)),
    ArcLengthContinuation(; predictor = :tangent)
)
@test SciMLBase.successful_retcode(s2)
@test s2.u[1] ≈ target atol = 1.0e-4
@test s2.u[2] ≈ s2.u[1]^2 atol = 1.0e-6   # stayed on the curve, no branch jump

prob32 = HomotopyProblem(Hm, Float32[4.0], (c = 4.0f0,); λspan = (0.0f0, 1.0f0))
s32 = solve(prob32, ArcLengthContinuation(; predictor = :tangent))
@test SciMLBase.successful_retcode(s32)
@test eltype(s32.u) == Float32           # no promotion through the Jacobian/null-space
@test s32.u[1] ≈ 2.0f0 atol = 1.0f-4
