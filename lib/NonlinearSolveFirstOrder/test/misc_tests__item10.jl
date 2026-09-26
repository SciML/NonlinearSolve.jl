using NonlinearSolveFirstOrder
using SciMLBase
using StaticArrays

alg = BoundedTrustRegion()
@test SciMLBase.allowsbounds(alg)
@test !SciMLBase.allowsbounds(TrustRegion())

interior_prob = NonlinearProblem(
    (u, p) -> u .^ 2 .- 4 .* u .+ 3, [1.5], nothing; lb = [0.0], ub = [2.0]
)
interior_sol = solve(interior_prob, alg)
@test SciMLBase.successful_retcode(interior_sol)
@test interior_sol.u ≈ [1.0] atol = 1.0e-10
@test all(interior_prob.lb .<= interior_sol.u .<= interior_prob.ub)

static_prob = NonlinearProblem(
    (u, p) -> u .^ 2 .- 4 .* u .+ 3, SVector(1.5), nothing;
    lb = SVector(0.0), ub = SVector(2.0)
)
static_sol = solve(static_prob, alg)
@test SciMLBase.successful_retcode(static_sol)
@test static_sol.u ≈ SVector(1.0) atol = 1.0e-10

exact_bound_nlls = NonlinearLeastSquaresProblem(
    (u, p) -> [u[1] - 2, 1.0], [0.0], nothing; lb = [-1.0], ub = [1.0]
)
bound_nlls_sol = solve(exact_bound_nlls, alg)
@test bound_nlls_sol.retcode == ReturnCode.Success
@test bound_nlls_sol.u == [1.0]
@test bound_nlls_sol.resid == [-1.0, 1.0]

function exact_bound_f!(resid, u, p)
    resid[1] = u[1] - 2
    resid[2] = 1
    return nothing
end
exact_bound_iip = NonlinearLeastSquaresProblem(
    NonlinearFunction(exact_bound_f!; resid_prototype = zeros(2)), [0.0], nothing;
    lb = [-1.0], ub = [1.0]
)
bound_iip_sol = solve(exact_bound_iip, alg)
@test bound_iip_sol.retcode == ReturnCode.Success
@test bound_iip_sol.u == [1.0]

no_root_prob = NonlinearProblem(
    (u, p) -> u .- 2, [0.0], nothing; lb = [-1.0], ub = [1.0]
)
no_root_sol = solve(no_root_prob, alg)
@test !SciMLBase.successful_retcode(no_root_sol)
@test no_root_sol.u == [1.0]
@test no_root_sol.resid == [-1.0]

partial_bounds_prob = NonlinearLeastSquaresProblem(
    (u, p) -> [u[1] - 2, u[2] - 3, 1.0], [0.0, 0.0], nothing;
    lb = nothing, ub = 1.0
)
partial_bounds_sol = solve(partial_bounds_prob, alg)
@test partial_bounds_sol.retcode == ReturnCode.Success
@test partial_bounds_sol.u == [1.0, 1.0]

function issue_1147_f!(resid, u, p)
    resid[1] = cos(u[2]) + sin(u[1]) - 0.5
    resid[2] = sin(u[2]) + cos(u[1]) - 0.3
    return resid
end

issue_1147_prob = NonlinearProblem(
    NonlinearFunction(issue_1147_f!), [0.3, 5.0], nothing;
    lb = [-100.0, 0.0], ub = [100.0, 10.0]
)
issue_1147_sol = solve(issue_1147_prob, alg)
@test SciMLBase.successful_retcode(issue_1147_sol)
@test maximum(abs, issue_1147_sol.resid) < 1.0e-10
@test all(issue_1147_prob.lb .<= issue_1147_sol.u .<= issue_1147_prob.ub)

calls = Ref(0)
function counted_f(u, p)
    calls[] += 1
    return u .- 1
end
infeasible_prob = NonlinearProblem(
    counted_f, [2.0], nothing; lb = [0.0], ub = [1.0]
)
@test_throws ArgumentError init(infeasible_prob, alg)
@test calls[] == 0

cache = init(interior_prob, BoundedTrustRegion(; initial_trust_radius = 1 // 4))
solve!(cache)
reinit!(cache, [1.5])
@test cache.trustregion_cache.trust_region == 0.25
@test_throws ArgumentError reinit!(cache, [3.0])
