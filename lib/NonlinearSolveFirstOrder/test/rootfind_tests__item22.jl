using NonlinearSolveFirstOrder, SciMLBase, StaticArrays, LinearSolve

# Powell's badly scaled function, and the same system in coordinates `y = s .* x`
powell(x, p) = [1.0e4 * x[1] * x[2] - 1, exp(-x[1]) + exp(-x[2]) - 1.0001]
const s = [1.0e3, 1.0e-2]
rescaled(y, p) = powell(y ./ s, p)
x0 = [0.0, 1.0]

autoscaled(; kwargs...) = TrustRegion(;
    subproblem = TrustRegionSubproblem.Dogleg,
    radius_update_scheme = RadiusUpdateSchemes.NLsolve, autoscale = true,
    linsolve = LUFactorization(), kwargs...
)

sol_x = solve(NonlinearProblem(powell, x0), autoscaled(); abstol = 1.0e-10)
sol_y = solve(NonlinearProblem(rescaled, s .* x0), autoscaled(); abstol = 1.0e-10)
@test SciMLBase.successful_retcode(sol_x)
@test SciMLBase.successful_retcode(sol_y)
# Column scaling makes the dogleg path, the radius and every accept decision invariant
# under `x ↦ s .* x`: both solves take the same steps
@test sol_y.u ≈ s .* sol_x.u rtol = 1.0e-6
@test sol_y.stats.nsteps == sol_x.stats.nsteps

# In place, and on static arrays, the iterates match the out-of-place solve
powell!(du, x, p) = (du .= powell(x, p); nothing)
sol_iip = solve(NonlinearProblem(powell!, copy(x0)), autoscaled(); abstol = 1.0e-10)
@test sol_iip.u ≈ sol_x.u rtol = 1.0e-10
@test sol_iip.stats.nsteps == sol_x.stats.nsteps
static_powell(x, p) = SVector{2}(powell(x, p))
sol_static = solve(NonlinearProblem(static_powell, SVector{2}(x0)), autoscaled(); abstol = 1.0e-10)
@test sol_static.u ≈ sol_x.u rtol = 1.0e-10

# A scalar problem scales by `|J|`, which leaves the dogleg on the Newton step
sol_scalar = solve(NonlinearProblem((x, p) -> x^3 - 8, 1.0), autoscaled())
@test sol_scalar.u ≈ 2

# Only a descent that measures a scaled region can run under `autoscale`
@test_throws ErrorException solve(
    NonlinearProblem(powell, x0),
    TrustRegion(; subproblem = TrustRegionSubproblem.More, autoscale = true)
)
