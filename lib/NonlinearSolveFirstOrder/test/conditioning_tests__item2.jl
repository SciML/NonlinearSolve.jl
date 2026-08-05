using NonlinearSolveFirstOrder, NonlinearSolveBase, SciMLBase, StaticArrays

# PCNR-style iterate limiting (Aadithya, Keiter & Mei): a voltage source, resistor and
# diode in the augmented unknowns [v, vj], where the junction voltage is an explicit
# unknown tied to v by a consistency equation. The `postcondition` option applies SPICE
# `pnjlim` limiting as the corrector; the framework re-evaluates the residual at the
# corrected iterate, which is PCNR's consistency property.
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
    r[1] = (u[1] - p.Vs) / p.R + p.Is * expm1(u[2] / p.Vt)
    r[2] = u[2] - u[1]
    return nothing
end
H! = (up, uprev, p) -> (up[2] = pnjlim(up[2], uprev[2], p.Vt, vcrit); nothing)
resid_norm(u) = (r = zeros(2); circuit!(r, u, cp); maximum(abs, r))

prob = NonlinearProblem(NonlinearFunction(circuit!), zeros(2), cp)
sol_plain = solve(prob, NewtonRaphson(); maxiters = 1000)
sol_lim = solve(prob, NewtonRaphson(); postcondition = H!, maxiters = 1000)
@test SciMLBase.successful_retcode(sol_lim)
@test resid_norm(sol_lim.u) < 1.0e-8
@test sol_lim.stats.nsteps < sol_plain.stats.nsteps ÷ 4

for alg in (NewtonRaphson(), TrustRegion(), LevenbergMarquardt())
    sol = solve(prob, alg; postcondition = H!, maxiters = 1000)
    @test SciMLBase.successful_retcode(sol)
    @test resid_norm(sol.u) < 1.0e-8
end

# carried on the problem rather than passed at solve time
prob_k = NonlinearProblem(NonlinearFunction(circuit!), zeros(2), cp; postcondition = H!)
@test solve(prob_k, NewtonRaphson(); maxiters = 1000).stats.nsteps == sol_lim.stats.nsteps

# init/solve! caching interface
cache = init(prob, NewtonRaphson(); postcondition = H!)
sol_cache = solve!(cache)
@test SciMLBase.successful_retcode(sol_cache) && resid_norm(sol_cache.u) < 1.0e-8

# four-argument correctors receive the solver cache (`nothing` for the initial-guess
# correction, which runs before a cache exists); only public accessors are used
cache_types = Set{Any}()
H4! = function (up, uprev, p, c)
    push!(cache_types, c === nothing ? Nothing : typeof(c))
    if c === nothing || NonlinearSolveBase.get_nsteps(c) < 100
        up[2] = pnjlim(up[2], uprev[2], p.Vt, vcrit)
    end
    return nothing
end
sol4 = solve(prob, NewtonRaphson(); postcondition = H4!, maxiters = 1000)
@test SciMLBase.successful_retcode(sol4) && resid_norm(sol4.u) < 1.0e-8
@test Nothing in cache_types
@test any(T -> T <: NonlinearSolveBase.AbstractNonlinearSolveCache, cache_types)

# out-of-place StaticArrays form, combining selective compression with limiting
f_s = (u, p) -> SA[(u[1] - p.Vs) / p.R + p.Is * expm1(u[2] / p.Vt), u[2] - u[1]]
sol_s = solve(
    NonlinearProblem(f_s, SA[0.0, 0.0], cp), NewtonRaphson();
    precondition = (fu, u, p) -> SA[asinh(fu[1]), fu[2]],
    postcondition = (up, uprev, p) -> SA[up[1], pnjlim(up[2], uprev[2], p.Vt, vcrit)],
    maxiters = 1000
)
@test SciMLBase.successful_retcode(sol_s) && resid_norm(Vector(sol_s.u)) < 1.0e-8

# projection-style corrector pinning a component exactly
fproj! = (du, u, p) -> (du[1] = u[1] - 1; du[2] = u[2]^2 - u[1] - 3; nothing)
sol_proj = solve(
    NonlinearProblem(fproj!, [5.0, 5.0]), NewtonRaphson();
    postcondition = (up, uprev, p) -> (up[1] = 1.0; nothing)
)
@test SciMLBase.successful_retcode(sol_proj)
@test sol_proj.u[1] == 1.0 && abs(sol_proj.u[2] - 2.0) < 1.0e-8

# bounds and the corrector are mutually exclusive
prob_b = NonlinearProblem(
    NonlinearFunction(circuit!), zeros(2), cp; lb = [-10.0, -10.0], ub = [10.0, 10.0]
)
@test_throws ArgumentError solve(prob_b, NewtonRaphson(); postcondition = H!)
