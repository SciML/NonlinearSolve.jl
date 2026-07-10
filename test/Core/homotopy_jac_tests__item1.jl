using NonlinearSolve

using SciMLBase
using LinearAlgebra

# Derivative-field passthrough: the sweep drivers must forward `jac` (λ-fixed to the
# standard 2/3-argument form), `jac_prototype`, `sparsity`, and `colorvec` from the
# problem's NonlinearFunction into the inner solver's NonlinearFunction instead of
# dropping them.
#
# The extra λ-free jac methods below exist only so the user-side NonlinearFunction
# constructor's arity-based iip conformance check accepts the λ-extended jac (MTK-style
# generated functions carry both dispatches naturally); they error loudly because the
# sweep must only ever call the λ-extended form through its own λ-fixing wrapper.

# f(u,p,λ) = (1-λ)*(u-c) + λ*(u^3-c); λ=0 root u=c ; λ=1 root u=∛c.
f_oop(u, p, λ) = [(1 - λ) * (u[1] - p) + λ * (u[1]^3 - p)]
f_iip(du, u, p, λ) = (du[1] = (1 - λ) * (u[1] - p) + λ * (u[1]^3 - p); nothing)

# --- Analytic oop jac is consumed (counting) ---
oop_jac_calls = Ref(0)
j_oop(u, p, λ) = (oop_jac_calls[] += 1; reshape([(1 - λ) + λ * 3 * u[1]^2;;], 1, 1))
j_oop(u, p) = error("λ-free jac must never be called by the sweep")
prob_oop = HomotopyProblem(NonlinearFunction{false}(f_oop; jac = j_oop), [2.0], 2.0)
sol_oop = solve(prob_oop, HomotopySweep(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(sol_oop)
@test sol_oop.u[1] ≈ cbrt(2.0) atol = 1.0e-8
@test oop_jac_calls[] > 0

# --- Analytic iip jac is consumed (counting) ---
iip_jac_calls = Ref(0)
function j_iip(J, u, p, λ)
    iip_jac_calls[] += 1
    J[1, 1] = (1 - λ) + λ * 3 * u[1]^2
    return nothing
end
j_iip(J, u, p) = error("λ-free jac must never be called by the sweep")
prob_iip = HomotopyProblem(NonlinearFunction{true}(f_iip; jac = j_iip), [2.0], 2.0)
sol_iip = solve(prob_iip, HomotopySweep(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(sol_iip)
@test sol_iip.u[1] ≈ cbrt(2.0) atol = 1.0e-8
@test iip_jac_calls[] > 0

# --- The default polyalgorithm sees the analytic jac (must_use_jacobian path) ---
polyalg_jac_calls = Ref(0)
j_poly(u, p, λ) = (polyalg_jac_calls[] += 1; reshape([(1 - λ) + λ * 3 * u[1]^2;;], 1, 1))
j_poly(u, p) = error("λ-free jac must never be called by the sweep")
prob_poly = HomotopyProblem(NonlinearFunction{false}(f_oop; jac = j_poly), [2.0], 2.0)
sol_poly = solve(prob_poly, HomotopySweep())
@test SciMLBase.successful_retcode(sol_poly)
@test sol_poly.u[1] ≈ cbrt(2.0) atol = 1.0e-8
@test polyalg_jac_calls[] > 0

# --- jac correctness: analytic jac reproduces the AD solution ---
prob_ad = HomotopyProblem(NonlinearFunction{false}(f_oop), [2.0], 2.0)
sol_ad = solve(prob_ad, HomotopySweep(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(sol_ad)
@test sol_oop.u[1] ≈ sol_ad.u[1] atol = 1.0e-12

# --- SimpleHomotopySweep consumes an analytic jac (SimpleNewtonRaphson honors f.jac) ---
simple_oop_jac_calls = Ref(0)
function j_soop(u, p, λ)
    simple_oop_jac_calls[] += 1
    return reshape([(1 - λ) + λ * 3 * u[1]^2;;], 1, 1)
end
j_soop(u, p) = error("λ-free jac must never be called by the sweep")
prob_soop = HomotopyProblem(NonlinearFunction{false}(f_oop; jac = j_soop), [2.0], 2.0)
sol_soop = solve(prob_soop, SimpleHomotopySweep(; inner = SimpleNewtonRaphson()))
@test SciMLBase.successful_retcode(sol_soop)
@test sol_soop.u[1] ≈ cbrt(2.0) atol = 1.0e-8
@test simple_oop_jac_calls[] > 0

simple_iip_jac_calls = Ref(0)
function j_siip(J, u, p, λ)
    simple_iip_jac_calls[] += 1
    J[1, 1] = (1 - λ) + λ * 3 * u[1]^2
    return nothing
end
j_siip(J, u, p) = error("λ-free jac must never be called by the sweep")
prob_siip = HomotopyProblem(NonlinearFunction{true}(f_iip; jac = j_siip), [2.0], 2.0)
sol_siip = solve(prob_siip, SimpleHomotopySweep(; inner = SimpleNewtonRaphson()))
@test SciMLBase.successful_retcode(sol_siip)
@test sol_siip.u[1] ≈ cbrt(2.0) atol = 1.0e-8
@test simple_iip_jac_calls[] > 0

# --- Tridiagonal jac_prototype: the inner solver's J is the structured prototype ---
# (1-λ)*(uᵢ-c) + λ*(3uᵢ - uᵢ₋₁ - uᵢ₊₁ + uᵢ³ - c) with zero boundary neighbors:
# diagonally dominant, tridiagonal Jacobian.
n = 50
function f_band(du, u, p, λ)
    for i in 1:length(u)
        um = i == 1 ? zero(eltype(u)) : u[i - 1]
        up = i == length(u) ? zero(eltype(u)) : u[i + 1]
        du[i] = (1 - λ) * (u[i] - p) + λ * (3u[i] - um - up + u[i]^3 - p)
    end
    return nothing
end
band_J_type = Ref{Any}(nothing)
function j_band(J, u, p, λ)
    band_J_type[] = typeof(J)
    for i in 1:length(u)
        J[i, i] = (1 - λ) + λ * (3 + 3u[i]^2)
        i > 1 && (J[i, i - 1] = -λ)
        i < length(u) && (J[i, i + 1] = -λ)
    end
    return nothing
end
j_band(J, u, p) = error("λ-free jac must never be called by the sweep")
proto = Tridiagonal(zeros(n - 1), ones(n), zeros(n - 1))
nf_band = NonlinearFunction{true}(f_band; jac = j_band, jac_prototype = proto)
prob_band = HomotopyProblem(nf_band, fill(1.0, n), 1.0)
sol_band = solve(prob_band, HomotopySweep(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(sol_band)
@test band_J_type[] <: Tridiagonal
resid_band = zeros(n)
f_band(resid_band, sol_band.u, 1.0, 1.0)
@test maximum(abs, resid_band) < 1.0e-8

# --- jac_prototype alone (no analytic jac) still passes through and solves ---
nf_proto = NonlinearFunction{true}(
    f_band; jac_prototype = Tridiagonal(zeros(n - 1), ones(n), zeros(n - 1))
)
prob_proto = HomotopyProblem(nf_proto, fill(1.0, n), 1.0)
sol_proto = solve(prob_proto, HomotopySweep(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(sol_proto)
resid_proto = zeros(n)
f_band(resid_proto, sol_proto.u, 1.0, 1.0)
@test maximum(abs, resid_proto) < 1.0e-8

# --- No derivative fields: the inner function is constructed exactly as before ---
import NonlinearSolveBase
fixλ = NonlinearSolveBase.FixLambda(prob_ad.f, 0.0)
fλ_plain = NonlinearSolveBase._sweep_nonlinear_function(Val(false), prob_ad.f, fixλ)
@test typeof(fλ_plain) == typeof(SciMLBase.NonlinearFunction{false}(fixλ))
@test fλ_plain.jac === nothing
@test fλ_plain.jac_prototype === nothing

# --- No-field problems solve identically (deterministic result + residual count) ---
resid_calls = Ref(0)
f_cnt(u, p, λ) = (resid_calls[] += 1; [(1 - λ) * (u[1] - p) + λ * (u[1]^3 - p)])
prob_cnt = HomotopyProblem(NonlinearFunction{false}(f_cnt), [2.0], 2.0)
sol_cnt1 = solve(prob_cnt, HomotopySweep(; inner = NewtonRaphson()))
calls1 = resid_calls[]
resid_calls[] = 0
sol_cnt2 = solve(prob_cnt, HomotopySweep(; inner = NewtonRaphson()))
@test SciMLBase.successful_retcode(sol_cnt1)
@test sol_cnt1.u == sol_cnt2.u
@test sol_cnt1.u == sol_ad.u
@test calls1 == resid_calls[]
@test calls1 > 0
