using NonlinearSolveFirstOrder, ADTypes, LineSearch

# The residual writes to a Float64 scratch buffer on the primal path (like a
# PreallocationTools.DiffCache), which ForwardDiff handles but Enzyme rejects
# when the function is passed as `Const`. The line search must therefore use
# the explicitly requested forward-mode backend, and loading Enzyme must not
# switch it to a reverse-mode one.
struct CachedResidual
    cache::Vector{Float64}
end
function (f::CachedResidual)(res, u, p)
    tmp = eltype(u) === Float64 ? f.cache : similar(u)
    @. tmp = u^2 - p
    @. res = tmp - u + 1.0
    return nothing
end

n = 4
u0 = ones(n)
p = fill(2.0, n)
prob = NonlinearLeastSquaresProblem(
    NonlinearFunction(CachedResidual(zeros(n)); resid_prototype = zeros(n)), u0, p
)

alg = GaussNewton(; linesearch = BackTracking(), autodiff = AutoForwardDiff())
sol = solve(prob, alg)
@test SciMLBase.successful_retcode(sol)

if isempty(VERSION.prerelease) && VERSION < v"1.12"
    using Enzyme

    sol = solve(prob, alg)
    @test SciMLBase.successful_retcode(sol)
end

struct AnalyticJacResidual end
function (::AnalyticJacResidual)(res, u, p)
    eltype(u) === Float64 || error("residual must not be differentiated")
    @. res = u^2 - p
    return nothing
end
function analytic_jacobian!(J, u, p)
    fill!(J, 0)
    for i in eachindex(u)
        J[i, i] = 2 * u[i]
    end
    return nothing
end

analytic_prob = NonlinearLeastSquaresProblem(
    NonlinearFunction(
        AnalyticJacResidual(); jac = analytic_jacobian!, jac_prototype = zeros(n, n),
        resid_prototype = zeros(n)
    ), u0, p
)
analytic_sol = solve(analytic_prob, GaussNewton(; linesearch = BackTracking()))
@test SciMLBase.successful_retcode(analytic_sol)
@test analytic_sol.u ≈ sqrt.(p)
