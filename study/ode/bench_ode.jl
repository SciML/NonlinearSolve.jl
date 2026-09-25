# Event location on an ODE interpolant, as DiffEqBase's `find_root` does for a
# ContinuousCallback: the condition g(u(t)) = |u(t)|² - r (the jump condition of quantum
# trajectories) on the Tsit5 interpolant of one step, abstol = 0.
#   julia --project=study/ode study/ode/bench_ode.jl
# Sources of the derivative:
#   rhs:    ∇g ⋅ f(u(t)), with f the ODE right-hand side at the interpolated state, as
#           OrdinaryDiffEq PR 4624 does; the state of the last condition evaluation is reused
#           when `jac` is called at the same t.
#   interp: ∇g ⋅ u'(t) from the derivative of the interpolant, `integrator(du, t, Val{1})`.
#   AD:     ForwardDiff through the interpolant, t ↦ g(integrator(t)) (out of place).
using OrdinaryDiffEqTsit5, BracketingNonlinearSolve, ForwardDiff, BenchmarkTools
using LinearAlgebra, Random, Printf
using ADTypes: AutoForwardDiff
include("../pr1330_newton_bisection.jl")

function make_steps(A; nsteps = 40, seed = 1, tol = 1.0e-6)
    rng = Random.Xoshiro(seed)
    n = size(A, 1)
    rhs!(du, u, p, t) = mul!(du, p, u)
    u0 = normalize(randn(rng, n))
    integ = init(
        ODEProblem(rhs!, u0, (0.0, 1.0e6), A), Tsit5();
        abstol = tol, reltol = tol, save_everystep = false
    )
    steps = []
    for _ in 1:nsteps
        step!(integ)
        g0, g1 = sum(abs2, integ.uprev), sum(abs2, integ.u)
        r = g0 + rand(rng) * (g1 - g0)   # a root inside the step
        push!(steps, (deepcopy(integ), r))
    end
    return steps
end

mutable struct Cond{I, T, M}
    integ::I
    r::Float64
    tmp::Vector{Float64}
    dtmp::Vector{Float64}
    t_tmp::Float64          # time of the state in tmp
    A::M
    counts::T
end

function Cond(integ, r, A)
    n = length(integ.u)
    return Cond(integ, r, zeros(n), zeros(n), NaN, A, zeros(Int, 3))
end

# as DiffEqBase's get_condition: the ends of the step are stored, not interpolated
function state!(c::Cond, t)
    t == c.integ.t && return c.integ.u
    t == c.integ.tprev && return c.integ.uprev
    if t != c.t_tmp
        c.integ(c.tmp, t)
        c.t_tmp = t
        c.counts[3] += 1   # interpolations
    end
    return c.tmp
end

function (c::Cond)(t, p)
    c.counts[1] += 1
    u = state!(c, t)
    return dot(u, u) - c.r
end

function jac_rhs(c::Cond, t)
    c.counts[2] += 1
    u = state!(c, t)
    mul!(c.dtmp, c.A, u)
    return 2dot(u, c.dtmp)
end

function jac_interp(c::Cond, t)
    c.counts[2] += 1
    u = state!(c, t)
    c.integ(c.dtmp, t, Val{1})
    return 2dot(u, c.dtmp)
end

# out-of-place, so a Dual t goes through the interpolant
function cond_oop(c::Cond, t)
    c.counts[t isa ForwardDiff.Dual ? 2 : 1] += 1
    u = c.integ(t)
    return dot(u, u) - c.r
end

function variants(c)
    f = (t, p) -> c(t, p)
    return [
        ("ModAB", IntervalNonlinearFunction(f), ModAB()),
        ("PR NewtonBisection, rhs", IntervalNonlinearFunction(f; jac = (t, p) -> jac_rhs(c, t)), PRNewtonBisection()),
        ("NewtonBisection, rhs", IntervalNonlinearFunction(f; jac = (t, p) -> jac_rhs(c, t)), NewtonBisection()),
        ("ModABNewton, rhs", IntervalNonlinearFunction(f; jac = (t, p) -> jac_rhs(c, t)), ModABNewton()),
        ("PR NewtonBisection, interp", IntervalNonlinearFunction(f; jac = (t, p) -> jac_interp(c, t)), PRNewtonBisection()),
        ("NewtonBisection, interp", IntervalNonlinearFunction(f; jac = (t, p) -> jac_interp(c, t)), NewtonBisection()),
        ("ModABNewton, interp", IntervalNonlinearFunction(f; jac = (t, p) -> jac_interp(c, t)), ModABNewton()),
        ("ModAB, out of place", IntervalNonlinearFunction((t, p) -> cond_oop(c, t)), ModAB()),
        ("NewtonBisection, AD", IntervalNonlinearFunction((t, p) -> cond_oop(c, t)), NewtonBisection(; autodiff = AutoForwardDiff())),
        ("ModABNewton, AD", IntervalNonlinearFunction((t, p) -> cond_oop(c, t)), ModABNewton(; autodiff = AutoForwardDiff())),
    ]
end

function run(name, A; tol = 1.0e-6)
    steps = make_steps(A; tol)
    conds = [Cond(integ, r, A) for (integ, r) in steps]
    nv = length(variants(conds[1]))
    println("\n### $name, Tsit5 with abstol = reltol = $tol\n")
    t_f = @belapsed $(conds[1])(0.5 * ($(steps[1][1]).tprev + $(steps[1][1]).t), nothing) setup = ($(conds[1]).t_tmp = NaN) evals = 1
    t_rhs = @belapsed mul!($(conds[1]).dtmp, $A, $(conds[1]).tmp)
    t_i1 = @belapsed $(steps[1][1])($(conds[1]).dtmp, 0.5 * ($(steps[1][1]).tprev + $(steps[1][1]).t), Val{1})
    @printf("condition (interpolation + g): %.0f ns; rhs mat-vec: %.0f ns; interpolant derivative: %.0f ns\n\n", t_f * 1.0e9, t_rhs * 1.0e9, t_i1 * 1.0e9)
    println("| method | f evals per event | jac or Dual evals per event | interpolations per event | μs per event | max abs(t - t_ModAB) |")
    println("|---|---|---|---|---|---|")
    ref = [solve(IntervalNonlinearProblem(IntervalNonlinearFunction((t, p) -> c(t, p)), (c.integ.tprev, c.integ.t)), ModAB(); abstol = 0.0).u for c in conds]
    for i in 1:nv
        vname = variants(conds[1])[i][1]
        tot = zeros(Int, 3)
        maxdev = 0.0
        for (k, c) in enumerate(conds)
            fill!(c.counts, 0)
            c.t_tmp = NaN
            _, fun, alg = variants(c)[i]
            sol = solve(IntervalNonlinearProblem(fun, (c.integ.tprev, c.integ.t)), alg; abstol = 0.0)
            SciMLBase.successful_retcode(sol) || println("  FAIL ", vname, " ", sol.retcode)
            maxdev = max(maxdev, abs(sol.u - ref[k]))
            tot .+= c.counts
        end
        probs = [(IntervalNonlinearProblem(variants(c)[i][2], (c.integ.tprev, c.integ.t)), variants(c)[i][3], c) for c in conds]
        t = @belapsed begin
            for (prob, alg, c) in $probs
                c.t_tmp = NaN
                solve(prob, alg; abstol = 0.0)
            end
        end samples = 300 evals = 1
        m = length(conds)
        @printf(
            "| %s | %.2f | %.2f | %.2f | %.2f | %.1e |\n", vname, tot[1] / m, tot[2] / m,
            tot[3] / m, t * 1.0e6 / m, maxdev
        )
    end
    return
end

# In place, as DiffEqBase evaluates conditions: a Dual time cannot be written into the
# Float64 cache.
function inplace_dual_check()
    A = [-0.1 1.0; -1.0 -0.1]
    (integ, r), = make_steps(A; nsteps = 1)
    c = Cond(integ, r, A)
    try
        solve(IntervalNonlinearProblem((t, p) -> c(t, p), (integ.tprev, integ.t)), NewtonBisection(; autodiff = AutoForwardDiff()); abstol = 0.0)
        println("\nIn-place condition with AD: works")
    catch e
        println("\nIn-place condition with AD: ", sprint(showerror, e; context = :limit => true)[1:min(end, 200)])
    end
    return
end

tridiag(n) = Tridiagonal(fill(1.0, n - 1), fill(-0.05, n), fill(-1.0, n - 1))
S = randn(Random.Xoshiro(2), 50, 50)
run("dense 50×50 generator (rhs costs more than the interpolation)", (S - S') / 2 - 0.05I)
run("tridiagonal 50×50 generator (rhs cheap)", tridiag(50))
run("tridiagonal 1000×1000 generator (rhs cheap)", tridiag(1000))
run("tridiagonal 50×50 generator (rhs cheap)", tridiag(50); tol = 1.0e-8)
inplace_dual_check()
