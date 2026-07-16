using NonlinearSolveFirstOrder, SciMLBase, StaticArrays

# Diode equation: Is*expm1(v/Vt) - It = 0. From v0 = 2 the exponential dominates and
# plain Newton creeps back with O(Vt)-sized steps, while the asinh-compressed residual
# G(F) = asinh(F) is nearly affine in v so Newton converges in a handful of steps.
p = (; Is = 1.0e-14, Vt = 0.025, It = 1.0e-2)
f_diode = (v, p) -> p.Is * expm1(v / p.Vt) - p.It
vstar = p.Vt * log(p.It / p.Is + 1)
G = (fu, u, p) -> asinh(fu)

sol_plain = solve(NonlinearProblem(f_diode, 2.0, p), NewtonRaphson(); maxiters = 200)
prob_G = NonlinearProblem(NonlinearFunction(f_diode; precondition = G), 2.0, p)
sol_G = solve(prob_G, NewtonRaphson(); maxiters = 200)
@test SciMLBase.successful_retcode(sol_G)
@test abs(sol_G.u - vstar) < 1.0e-8
@test sol_G.stats.nsteps < sol_plain.stats.nsteps ÷ 2

# in-place vector form
f_iip = (du, u, p) -> (du[1] = p.Is * expm1(u[1] / p.Vt) - p.It; nothing)
G_iip = (fu, u, p) -> (fu[1] = asinh(fu[1]); nothing)
prob_Gi = NonlinearProblem(NonlinearFunction(f_iip; precondition = G_iip), [2.0], p)
sol_Gi = solve(prob_Gi, NewtonRaphson(); maxiters = 200)
@test SciMLBase.successful_retcode(sol_Gi)
@test abs(sol_Gi.u[1] - vstar) < 1.0e-8

# StaticArrays out-of-place form
f_s = (u, p) -> SA[p.Is * expm1(u[1] / p.Vt) - p.It]
G_s = (fu, u, p) -> asinh.(fu)
prob_Gs = NonlinearProblem(NonlinearFunction(f_s; precondition = G_s), SA[2.0], p)
sol_Gs = solve(prob_Gs, NewtonRaphson(); maxiters = 200)
@test SciMLBase.successful_retcode(sol_Gs)
@test abs(sol_Gs.u[1] - vstar) < 1.0e-8

# termination is measured on the preconditioned residual and the reported resid is the
# preconditioned one
@test abs(sol_G.resid - asinh(f_diode(sol_G.u, p))) < 1.0e-12

# NonlinearLeastSquaresProblem with a residual re-weighting transform (for a consistent
# system any root-preserving G keeps the same solution)
f_nlls! = (r, u, p) -> (r[1] = u[1] - 1; r[2] = 10 * (u[1] - 1); r[3] = u[2] - 2; nothing)
G_nlls! = (fu, u, p) -> (fu[2] *= 0.1; nothing)
nlls = NonlinearLeastSquaresProblem(
    NonlinearFunction(f_nlls!; resid_prototype = zeros(3), precondition = G_nlls!),
    [0.0, 0.0]
)
sol_nlls = solve(nlls, GaussNewton())
@test SciMLBase.successful_retcode(sol_nlls)
@test isapprox(sol_nlls.u, [1.0, 2.0]; atol = 1.0e-6)
