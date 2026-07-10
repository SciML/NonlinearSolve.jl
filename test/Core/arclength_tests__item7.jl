using NonlinearSolve

using SciMLBase

const NLSB = NonlinearSolve.NonlinearSolveBase

# ---- theta kwarg: storage + validation (0 and 1 are excluded — either weight would
# vanish and the metric would degenerate) ----
alg = ArcLengthContinuation()
@test alg.theta ≈ 0.5

alg_θ = ArcLengthContinuation(; theta = 0.3)
@test alg_θ.theta ≈ 0.3

@test_throws ArgumentError ArcLengthContinuation(; theta = 0.0)
@test_throws ArgumentError ArcLengthContinuation(; theta = 1.0)
@test_throws ArgumentError ArcLengthContinuation(; theta = -0.5)
@test_throws ArgumentError ArcLengthContinuation(; theta = 1.5)

# ---- bordered tangent, unit-tested on known Jacobians (n = 1, θ = 0.5 ⇒ wu = wλ = 0.5)
wu, wλ = 0.5, 0.5

# Regular point of u^3 - 3u - (-3 + 6λ) at (u, λ) = (-2.1038034, 0):
# J = [3u^2 - 3, -6]; the analytic null direction is [6, 3u^2 - 3].
u0pt = -2.1038034
J_reg = [3 * u0pt^2 - 3 -6.0]
τprev = [0.0, 1.0 / sqrt(wλ)]                       # the driver's pure-λ seed, θ-unit
# the in-place API takes caller-preallocated scratch (B overwritten by lu!, t by the
# solution) and returns t
bord!(J, τp) = copy(
    NLSB._bordered_tangent!(
        Matrix{Float64}(undef, 2, 2), Vector{Float64}(undef, 2), J, τp, wu, wλ, 1
    )
)
τ_reg = bord!(J_reg, τprev)
@test abs(J_reg[1] * τ_reg[1] + J_reg[2] * τ_reg[2]) < 1.0e-12   # J τ = 0
@test wu * τ_reg[1]^2 + wλ * τ_reg[2]^2 ≈ 1.0                    # θ-unit
@test wλ * τprev[2] * τ_reg[2] > 0                               # oriented along τprev
d_analytic = [6.0, 3 * u0pt^2 - 3]
d_analytic ./= sqrt(wu * d_analytic[1]^2 + wλ * d_analytic[2]^2)
@test τ_reg ≈ d_analytic atol = 1.0e-12

# Fold point (u, λ) = (1, 1/6): J = [0, -6], so the tangent must be vertical in λ.
J_fold = [0.0 -6.0]
τ_fold = bord!(J_fold, [1.0, 0.5])
@test abs(τ_fold[2]) < 1.0e-12                       # vertical in λ
@test wu * τ_fold[1]^2 ≈ 1.0                         # θ-unit
@test τ_fold[1] > 0                                  # oriented along τprev = [1, 0.5]

# SVD fallback: τprev θ-orthogonal to the tangent at the fold makes the bordered matrix
# exactly singular ([J; wᵀ] has a zero first column); the fallback must return the same
# (θ-unit, vertical-in-λ) tangent without erroring.
τ_fb = bord!(J_fold, [0.0, 1.0 / sqrt(wλ)])
@test abs(τ_fb[2]) < 1.0e-12
@test wu * τ_fb[1]^2 ≈ 1.0

# ---- S-fold end-to-end: the bordered :tangent predictor still rounds both folds to the
# connected upper-sheet root ----
target = 2.1038034
Hfold(u, p, λ) = [u[1]^3 - 3 * u[1] - (-3 + 6λ)]
prob = HomotopyProblem(Hfold, [-target]; λspan = (0.0, 1.0))
sol_t = solve(prob, ArcLengthContinuation(; predictor = :tangent))
@test SciMLBase.successful_retcode(sol_t)
@test sol_t.u[1] ≈ target atol = 1.0e-4
@test abs(sol_t.u[1]^3 - 3 * sol_t.u[1] - 3) < 1.0e-6

# default θ matches the secant predictor's root (values, not step counts)
sol_s = solve(prob, ArcLengthContinuation(; predictor = :secant))
@test sol_t.u[1] ≈ sol_s.u[1] atol = 1.0e-4

# a non-default θ still lands on the same connected-branch root
sol_θ = solve(prob, ArcLengthContinuation(; predictor = :tangent, theta = 0.8))
@test SciMLBase.successful_retcode(sol_θ)
@test sol_θ.u[1] ≈ target atol = 1.0e-4

# ---- SVD fallback end-to-end: fold exactly at the start point. On u^2 - λ = 0 from
# (u, λ) = (0, 0) the first tangent meets the fold head-on with the pure-λ seed as
# bordering row — the singular bordered solve must hand off to the SVD and the
# continuation must still reach λ = 1 (u = ±1; the fallback's orientation is free, and
# either sheet of the parabola reaches the target) ----
Hpar(u, p, λ) = [u[1]^2 - λ]
prob_par = HomotopyProblem(Hpar, [0.0]; λspan = (0.0, 1.0))
sol_par = solve(prob_par, ArcLengthContinuation(; predictor = :tangent))
@test SciMLBase.successful_retcode(sol_par)
@test abs(sol_par.u[1]) ≈ 1.0 atol = 1.0e-6
@test abs(sol_par.u[1]^2 - 1.0) < 1.0e-6

# ---- previous-roots regression on the standard monotone problem ----
Hm(u, p, λ) = [(1 - λ) * (u[1] - p.c) + λ * (u[1]^2 - p.c)]
sm = solve(HomotopyProblem(Hm, [4.0], (c = 4.0,)), ArcLengthContinuation(; predictor = :tangent))
@test SciMLBase.successful_retcode(sm)
@test sm.u[1] ≈ 2.0 atol = 1.0e-6
sm_sec = solve(HomotopyProblem(Hm, [4.0], (c = 4.0,)), ArcLengthContinuation())
@test SciMLBase.successful_retcode(sm_sec)
@test sm_sec.u[1] ≈ 2.0 atol = 1.0e-6

# ---- n = 50 fold-free coupled system: exercises the 1/n-scaled metric and the bordered
# solve at moderate n. M u + λ u.^3 = c with M = I + 0.25 (shift left + shift right)
# (diagonally dominant ⇒ nonsingular Jacobian along the whole path), c built so that
# u = ones(n) solves the λ = 1 system exactly ----
nbig = 50
function Hbig(u, p, λ)
    n = length(u)
    r = similar(u)
    for i in 1:n
        acc = u[i]
        i > 1 && (acc += 0.25 * u[i - 1])
        i < n && (acc += 0.25 * u[i + 1])
        r[i] = acc + λ * u[i]^3 - p.c[i]
    end
    return r
end
cbig = [1.0 + 0.25 * ((i > 1) + (i < nbig)) + 1.0 for i in 1:nbig]
prob_big = HomotopyProblem(Hbig, ones(nbig), (c = cbig,); λspan = (0.0, 1.0))
sol_big = solve(prob_big, ArcLengthContinuation(; predictor = :tangent))
@test SciMLBase.successful_retcode(sol_big)
@test maximum(abs, sol_big.u .- 1.0) < 1.0e-6
@test maximum(abs, Hbig(sol_big.u, (c = cbig,), 1.0)) < 1.0e-6

# ---- Float32 stays Float32 through the bordered solve and the θ-weighted metric ----
prob32 = HomotopyProblem(Hm, Float32[4.0], (c = 4.0f0,); λspan = (0.0f0, 1.0f0))
s32 = solve(prob32, ArcLengthContinuation(; predictor = :tangent))
@test SciMLBase.successful_retcode(s32)
@test eltype(s32.u) == Float32
@test s32.u[1] ≈ 2.0f0 atol = 1.0f-4
