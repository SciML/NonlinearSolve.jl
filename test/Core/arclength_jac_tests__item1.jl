using NonlinearSolve

using SciMLBase
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings

# Derivative-field support for ArcLengthContinuation (issue #1020 item 2, the follow-up
# left out of the sweep-driver PR): the user's λ-extended jac supplies only the n×n
# ∂H/∂u block, so the driver completes the augmented path Jacobian [∂H/∂u | ∂H/∂λ] with
# the ∂H/∂λ column as one scalar forward derivative, borders it with the analytically
# known θ-weighted Keller constraint row for the corrector, and extends
# jac_prototype/sparsity/colorvec to the augmented/bordered shapes.
#
# The extra λ-free jac methods below exist only so the user-side NonlinearFunction
# constructor's arity-based iip conformance check accepts the λ-extended jac (MTK-style
# generated functions carry both dispatches naturally); they error loudly because the
# driver must only ever call the λ-extended form through its own wrappers.

# S-fold: u^3 - 3u = -3 + 6λ; folds (∂H/∂u = 0) at u = ±1, i.e. λ = 5/6 and 1/6, so the
# path from the λ=0 lower root to the λ=1 upper root is non-monotone in λ — the analytic
# jac is consumed by the genuinely bordered machinery, not an easy λ-sweep.
target = 2.1038034
f_fold_oop(u, p, λ) = [u[1]^3 - 3 * u[1] - (-3 + 6λ)]
f_fold_iip(du, u, p, λ) = (du[1] = u[1]^3 - 3 * u[1] - (-3 + 6λ); nothing)

# --- Augmented corrector consumes the analytic oop jac (default :secant predictor) ---
oop_jac_λs = Float64[]
j_oop(u, p, λ) = (push!(oop_jac_λs, λ); reshape([3 * u[1]^2 - 3;;], 1, 1))
j_oop(u, p) = error("λ-free jac must never be called by arclength")
prob_oop = HomotopyProblem(
    NonlinearFunction{false}(f_fold_oop; jac = j_oop), [-target]; λspan = (0.0, 1.0)
)
sol_oop = solve(prob_oop, ArcLengthContinuation(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(sol_oop)
@test sol_oop.u[1] ≈ target atol = 1.0e-4
@test length(oop_jac_λs) > 0
# jac evaluations at interior λ can only come from the augmented corrector (the λ-fixed
# anchor and landing solves evaluate exactly at λ = 0 and λ = 1)
@test any(l -> 1.0e-3 < l < 1 - 1.0e-3, oop_jac_λs)

# --- Augmented corrector consumes the analytic iip jac ---
iip_jac_λs = Float64[]
function j_iip(J, u, p, λ)
    push!(iip_jac_λs, λ)
    J[1, 1] = 3 * u[1]^2 - 3
    return nothing
end
j_iip(J, u, p) = error("λ-free jac must never be called by arclength")
prob_iip = HomotopyProblem(
    NonlinearFunction{true}(f_fold_iip; jac = j_iip), [-target]; λspan = (0.0, 1.0)
)
sol_iip = solve(prob_iip, ArcLengthContinuation(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(sol_iip)
@test sol_iip.u[1] ≈ target atol = 1.0e-4
@test any(l -> 1.0e-3 < l < 1 - 1.0e-3, iip_jac_λs)

# --- The :tangent predictor's path Jacobian consumes the analytic jac ---
empty!(oop_jac_λs)
sol_tan = solve(
    prob_oop, ArcLengthContinuation(; inner = NewtonRaphson(), predictor = :tangent)
)
@test SciMLBase.successful_retcode(sol_tan)
@test sol_tan.u[1] ≈ target atol = 1.0e-4
@test length(oop_jac_λs) > 0

# --- jac correctness: the analytic-jac solution matches the AD solution ---
prob_ad = HomotopyProblem(NonlinearFunction{false}(f_fold_oop), [-target]; λspan = (0.0, 1.0))
sol_ad = solve(prob_ad, ArcLengthContinuation(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(sol_ad)
@test sol_oop.u[1] ≈ sol_ad.u[1] atol = 1.0e-6
sol_ad_tan = solve(
    prob_ad, ArcLengthContinuation(; inner = NewtonRaphson(), predictor = :tangent)
)
@test sol_tan.u[1] ≈ sol_ad_tan.u[1] atol = 1.0e-6

# --- Bordered prototype structure: user band + dense ∂H/∂λ column + dense constraint
# row, promoted from Tridiagonal to SparseMatrixCSC ---
import NonlinearSolveBase
n = 50
proto = Tridiagonal(zeros(n - 1), ones(n), zeros(n - 1))
aug_proto = NonlinearSolveBase._augmented_prototype(proto, n)
bord_proto = NonlinearSolveBase._bordered_prototype(proto, n)
@test aug_proto isa SparseMatrixCSC && size(aug_proto) == (n, n + 1)
@test bord_proto isa SparseMatrixCSC && size(bord_proto) == (n + 1, n + 1)
Is, Js, _ = findnz(bord_proto)
S = Set(zip(Is, Js))
@test all(((i, i + 1) in S) for i in 1:(n - 1))   # superdiagonal survived zero values
@test all(((i + 1, i) in S) for i in 1:(n - 1))   # subdiagonal survived zero values
@test all(((i, i) in S) for i in 1:n)             # diagonal
@test all(((i, n + 1) in S) for i in 1:(n + 1))   # dense ∂H/∂λ column + corner
@test all(((n + 1, j) in S) for j in 1:n)         # dense constraint row
# Diagonal is the type where plain `sparse` is value-based (`sparse(Diagonal(zeros(n)))`
# has zero stored entries) — the structural conversion must keep the full diagonal
diag_aug = NonlinearSolveBase._augmented_prototype(Diagonal(zeros(n)), n)
@test diag_aug isa SparseMatrixCSC && size(diag_aug) == (n, n + 1)
Id, Jd, _ = findnz(diag_aug)
Sd = Set(zip(Id, Jd))
@test all(((i, i) in Sd) for i in 1:n)            # diagonal survived zero values
@test all(((i, n + 1) in Sd) for i in 1:n)        # dense ∂H/∂λ column

# a SparseMatrixCSC prototype borders without conversion
csc_proto = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))
bord_csc = NonlinearSolveBase._bordered_prototype(csc_proto, n)
@test bord_csc isa SparseMatrixCSC && size(bord_csc) == (n + 1, n + 1)
@test bord_csc[n + 1, 1] == 1 && bord_csc[1, n + 1] == 1 && bord_csc[2, 1] == 1

# --- n = 50 fold-free banded system: analytic jac + Tridiagonal prototype, :tangent.
# (1-λ)*(uᵢ-c) + λ*(3uᵢ - uᵢ₋₁ - uᵢ₊₁ + uᵢ³ - c) with zero boundary neighbors:
# diagonally dominant, tridiagonal ∂H/∂u along the whole path (no folds). ---
function f_band(du, u, p, λ)
    for i in 1:length(u)
        um = i == 1 ? zero(eltype(u)) : u[i - 1]
        up = i == length(u) ? zero(eltype(u)) : u[i + 1]
        du[i] = (1 - λ) * (u[i] - p) + λ * (3u[i] - um - up + u[i]^3 - p)
    end
    return nothing
end
band_J_kinds = Set{Any}()
function j_band(J, u, p, λ)
    A = J
    while A isa SubArray
        A = parent(A)
    end
    push!(band_J_kinds, (typeof(A).name.wrapper, size(A)))
    for i in 1:length(u)
        J[i, i] = (1 - λ) + λ * (3 + 3u[i]^2)
        i > 1 && (J[i, i - 1] = -λ)
        i < length(u) && (J[i, i + 1] = -λ)
    end
    return nothing
end
j_band(J, u, p) = error("λ-free jac must never be called by arclength")
nf_band = NonlinearFunction{true}(f_band; jac = j_band, jac_prototype = proto)
prob_band = HomotopyProblem(nf_band, fill(1.0, n), 1.0)
sol_band = solve(
    prob_band, ArcLengthContinuation(; inner = NewtonRaphson(), predictor = :tangent)
)
@test SciMLBase.successful_retcode(sol_band)
resid_band = zeros(n)
f_band(resid_band, sol_band.u, 1.0, 1.0)
@test maximum(abs, resid_band) < 1.0e-8
# the corrector's Jacobian buffer is the bordered (n+1)×(n+1) SparseMatrixCSC — the
# user jac saw it through the driver's top-block view
@test (SparseMatrixCSC, (n + 1, n + 1)) in band_J_kinds
# the λ-fixed anchor/landing solves kept the user's own Tridiagonal prototype
@test (Tridiagonal, (n, n)) in band_J_kinds

# --- jac_prototype + colorvec WITHOUT an analytic jac: the packed predictor system
# sparse-ADs with the extended colorvec, the corrector sparse-ADs with the bordered
# pattern (its coloring recomputed — the dense constraint row admits no nontrivial
# column coloring), and everything still solves ---
band_colorvec = [mod1(i, 3) for i in 1:n]
nf_proto = NonlinearFunction{true}(
    f_band; jac_prototype = Tridiagonal(zeros(n - 1), ones(n), zeros(n - 1)),
    colorvec = band_colorvec
)
prob_proto = HomotopyProblem(nf_proto, fill(1.0, n), 1.0)
sol_proto = solve(
    prob_proto, ArcLengthContinuation(; inner = NewtonRaphson(), predictor = :tangent)
)
@test SciMLBase.successful_retcode(sol_proto)
resid_proto = zeros(n)
f_band(resid_proto, sol_proto.u, 1.0, 1.0)
@test maximum(abs, resid_proto) < 1.0e-8

# --- No derivative fields: the packed and augmented functions are constructed exactly
# as before ---
fplain = NonlinearFunction{false}(f_fold_oop)
g = NonlinearSolveBase.HomotopyResidual(fplain, 1)
pf = NonlinearSolveBase._arclength_path_function(Val(false), fplain, nothing, 1)
@test typeof(pf) == typeof(SciMLBase.NonlinearFunction{false}(g))
@test pf.jac === nothing && pf.jac_prototype === nothing
aug = NonlinearSolveBase.AugmentedHomotopy(fplain, [0.0, 1.0], [0.0, 0.0], 0.1, 1, 0.5, 0.5)
af = NonlinearSolveBase._arclength_augmented_function(
    Val(false), fplain, aug, nothing, nothing, nothing
)
@test typeof(af) == typeof(SciMLBase.NonlinearFunction{false}(aug))
@test af.jac === nothing && af.jac_prototype === nothing

# --- No-field problems solve identically (deterministic root + residual-eval count;
# the counts below were also verified unchanged against the pre-change base branch) ---
resid_calls = Ref(0)
f_cnt(u, p, λ) = (resid_calls[] += 1; [u[1]^3 - 3 * u[1] - (-3 + 6λ)])
prob_cnt = HomotopyProblem(NonlinearFunction{false}(f_cnt), [-target]; λspan = (0.0, 1.0))
sol_cnt1 = solve(prob_cnt, ArcLengthContinuation(; inner = NewtonRaphson()))
calls_secant = resid_calls[]
resid_calls[] = 0
sol_cnt2 = solve(prob_cnt, ArcLengthContinuation(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(sol_cnt1)
@test sol_cnt1.u == sol_cnt2.u
@test sol_cnt1.u[1] ≈ sol_ad.u[1] atol = 1.0e-10
@test calls_secant == resid_calls[]
@test calls_secant > 0
resid_calls[] = 0
sol_cnt3 = solve(
    prob_cnt, ArcLengthContinuation(; inner = NewtonRaphson(), predictor = :tangent)
)
calls_tangent = resid_calls[]
@test SciMLBase.successful_retcode(sol_cnt3)
@test sol_cnt3.u[1] ≈ sol_ad_tan.u[1] atol = 1.0e-10
@test calls_tangent > 0
