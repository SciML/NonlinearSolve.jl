using NonlinearSolveFirstOrder, SciMLBase, StaticArrays

# Diode equation: from v0 = 2 the exponential dominates and plain Newton creeps back in
# O(Vt)-sized steps, while the asinh-compressed residual is nearly affine in v.
p = (; Is = 1.0e-14, Vt = 0.025, It = 1.0e-2)
f_diode = (v, p) -> p.Is * expm1(v / p.Vt) - p.It
vstar = p.Vt * log(p.It / p.Is + 1)
G = (fu, u, p) -> asinh(fu)
prob = NonlinearProblem(f_diode, 2.0, p)

sol_plain = solve(prob, NewtonRaphson(); maxiters = 200)
sol_G = solve(prob, NewtonRaphson(); precondition = G, maxiters = 200)
@test SciMLBase.successful_retcode(sol_G)
@test abs(sol_G.u - vstar) < 1.0e-8
@test sol_G.stats.nsteps < sol_plain.stats.nsteps ÷ 2
# the option is late-bound: the problem is unchanged and re-solves as before
@test solve(prob, NewtonRaphson(); maxiters = 200).stats.nsteps == sol_plain.stats.nsteps

# carried on the problem instead, and overridden at solve time
prob_k = NonlinearProblem(f_diode, 2.0, p; precondition = G)
@test solve(prob_k, NewtonRaphson(); maxiters = 200).stats.nsteps == sol_G.stats.nsteps
@test solve(
    prob_k, NewtonRaphson(); precondition = (fu, u, p) -> fu, maxiters = 200
).stats.nsteps == sol_plain.stats.nsteps

# in-place vector and StaticArrays out-of-place forms
f_iip = (du, u, p) -> (du[1] = p.Is * expm1(u[1] / p.Vt) - p.It; nothing)
G_iip = (fu, u, p) -> (fu[1] = asinh(fu[1]); nothing)
sol_i = solve(
    NonlinearProblem(f_iip, [2.0], p), NewtonRaphson(); precondition = G_iip, maxiters = 200
)
@test SciMLBase.successful_retcode(sol_i) && abs(sol_i.u[1] - vstar) < 1.0e-8

f_s = (u, p) -> SA[p.Is * expm1(u[1] / p.Vt) - p.It]
sol_s = solve(
    NonlinearProblem(f_s, SA[2.0], p), NewtonRaphson();
    precondition = (fu, u, p) -> asinh.(fu), maxiters = 200
)
@test SciMLBase.successful_retcode(sol_s) && abs(sol_s.u[1] - vstar) < 1.0e-8

# termination and the reported residual are measured on the preconditioned map
@test abs(sol_G.resid - asinh(f_diode(sol_G.u, p))) < 1.0e-12

# NonlinearLeastSquaresProblem: a residual re-weighting keeps the solution of a
# consistent system
f_nlls! = (r, u, p) -> (r[1] = u[1] - 1; r[2] = 10 * (u[1] - 1); r[3] = u[2] - 2; nothing)
sol_nlls = solve(
    NonlinearLeastSquaresProblem(
        NonlinearFunction(f_nlls!; resid_prototype = zeros(3)), [0.0, 0.0]
    ),
    GaussNewton(); precondition = (fu, u, p) -> (fu[2] *= 0.1; nothing)
)
@test SciMLBase.successful_retcode(sol_nlls)
@test isapprox(sol_nlls.u, [1.0, 2.0]; atol = 1.0e-6)
