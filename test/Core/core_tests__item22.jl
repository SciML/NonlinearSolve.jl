using NonlinearSolve, SciMLBase

# Nonlinear preconditioning hooks through the default polyalgorithm and the
# unified `solve` entry points (SciML/NonlinearSolve.jl#351).
function pnjlim(vnew, vold, vt, vcrit)
    if vnew > vcrit && abs(vnew - vold) > 2vt
        if vold > 0
            arg = 1 + (vnew - vold) / vt
            vnew = arg > 0 ? vold + vt * log(arg) : vcrit
        else
            vnew = vt * log(vnew / vt)
        end
    end
    return vnew
end
cp = (; Vs = 5.0, R = 1.0e3, Is = 1.0e-14, Vt = 0.025)
vcrit = cp.Vt * log(cp.Vt / (sqrt(2) * cp.Is))
function circuit!(r, u, p)
    v, vj = u[1], u[2]
    r[1] = (v - p.Vs) / p.R + p.Is * expm1(u[2] / p.Vt)
    r[2] = vj - v
    return nothing
end
H! = (up, uprev, p) -> (up[2] = pnjlim(up[2], uprev[2], p.Vt, vcrit); nothing)
prob_lim = NonlinearProblem(NonlinearFunction(circuit!; postcondition = H!), zeros(2), cp)

sol = solve(prob_lim; maxiters = 1000)
@test SciMLBase.successful_retcode(sol)
r = zeros(2); circuit!(r, sol.u, cp)
@test maximum(abs, r) < 1.0e-8

# precondition through the default polyalgorithm
p = (; Is = 1.0e-14, Vt = 0.025, It = 1.0e-2)
f_diode = (v, p) -> p.Is * expm1(v / p.Vt) - p.It
G = (fu, u, p) -> asinh(fu)
sol_G = solve(NonlinearProblem(NonlinearFunction(f_diode; precondition = G), 2.0, p))
@test SciMLBase.successful_retcode(sol_G)
@test abs(sol_G.u - p.Vt * log(p.It / p.Is + 1)) < 1.0e-8
