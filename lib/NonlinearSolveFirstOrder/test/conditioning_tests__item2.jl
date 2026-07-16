using NonlinearSolveFirstOrder, NonlinearSolveBase, SciMLBase, StaticArrays

# PCNR-style iterate limiting (Aadithya, Keiter & Mei): a series voltage source,
# resistor, and diode in the augmented unknowns [v, vj], where the junction voltage vj
# is an explicit unknown tied to v by a consistency equation. The `postcondition`
# applies SPICE pnjlim limiting to vj as the corrector; the framework re-evaluates the
# residual at the corrected iterate so residual/Jacobian stay consistent (the PCNR
# consistency property).
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
    r[1] = (v - p.Vs) / p.R + p.Is * expm1(vj / p.Vt)
    r[2] = vj - v
    return nothing
end
H! = (up, uprev, p) -> (up[2] = pnjlim(up[2], uprev[2], p.Vt, vcrit); nothing)
resid_norm(u) = (r = zeros(2); circuit!(r, u, cp); maximum(abs, r))

prob_plain = NonlinearProblem(NonlinearFunction(circuit!), zeros(2), cp)
sol_plain = solve(prob_plain, NewtonRaphson(); maxiters = 1000)

prob_lim = NonlinearProblem(NonlinearFunction(circuit!; postcondition = H!), zeros(2), cp)
for alg in (NewtonRaphson(), TrustRegion(), LevenbergMarquardt())
    sol = solve(prob_lim, alg; maxiters = 1000)
    @test SciMLBase.successful_retcode(sol)
    @test resid_norm(sol.u) < 1.0e-8
end

sol_lim = solve(prob_lim, NewtonRaphson(); maxiters = 1000)
@test sol_lim.stats.nsteps < sol_plain.stats.nsteps ÷ 4

# init/solve! caching interface
cache = init(prob_lim, NewtonRaphson())
sol_cache = solve!(cache)
@test SciMLBase.successful_retcode(sol_cache)
@test resid_norm(sol_cache.u) < 1.0e-8

# out-of-place StaticArrays form combining selective residual compression with limiting
f_s = (u, p) -> SA[(u[1] - p.Vs) / p.R + p.Is * expm1(u[2] / p.Vt), u[2] - u[1]]
G_s = (fu, u, p) -> SA[asinh(fu[1]), fu[2]]
H_s = (up, uprev, p) -> SA[up[1], pnjlim(up[2], uprev[2], p.Vt, vcrit)]
fn_s = NonlinearFunction(f_s; precondition = G_s, postcondition = H_s)
sol_s = solve(NonlinearProblem(fn_s, SA[0.0, 0.0], cp), NewtonRaphson(); maxiters = 1000)
@test SciMLBase.successful_retcode(sol_s)
@test resid_norm(Vector(sol_s.u)) < 1.0e-8

# four-argument hook form: receives the solver cache (or nothing at the initial-guess
# correction); only public accessors are used
cache_types = Set{Any}()
H4! = function (up, uprev, p, cache)
    push!(cache_types, cache === nothing ? Nothing : typeof(cache))
    if cache === nothing || NonlinearSolveBase.get_nsteps(cache) < 100
        up[2] = pnjlim(up[2], uprev[2], p.Vt, vcrit)
    end
    return nothing
end
prob_lim4 = NonlinearProblem(NonlinearFunction(circuit!; postcondition = H4!), zeros(2), cp)
sol_lim4 = solve(prob_lim4, NewtonRaphson(); maxiters = 1000)
@test SciMLBase.successful_retcode(sol_lim4)
@test resid_norm(sol_lim4.u) < 1.0e-8
@test Nothing in cache_types
@test any(T -> T <: NonlinearSolveBase.AbstractNonlinearSolveCache, cache_types)

# postcondition that enforces a known solution component exactly (projection-style)
fproj! = (du, u, p) -> (du[1] = u[1] - 1; du[2] = u[2]^2 - u[1] - 3; nothing)
Hproj! = (up, uprev, p) -> (up[1] = 1.0; nothing)
prob_proj = NonlinearProblem(NonlinearFunction(fproj!; postcondition = Hproj!), [5.0, 5.0])
sol_proj = solve(prob_proj, NewtonRaphson())
@test SciMLBase.successful_retcode(sol_proj)
@test sol_proj.u[1] == 1.0
@test abs(sol_proj.u[2] - 2.0) < 1.0e-8

# bounds + postcondition is rejected
prob_bounds = NonlinearProblem(
    NonlinearFunction(circuit!; postcondition = H!), zeros(2), cp;
    lb = [-10.0, -10.0], ub = [10.0, 10.0]
)
@test_throws ArgumentError solve(prob_bounds, NewtonRaphson())
