using NonlinearSolveFirstOrder, NonlinearSolveBase, LinearAlgebra, StaticArrays,
    LinearSolve, StableRNGs

# Underdetermined NLLS (issue #851): LevenbergMarquardt should converge, and with the
# minimum-norm formulation it should find the minimum-norm solution when started at zero.

# Linear underdetermined: 2 equations, 4 unknowns; minimum-norm solution from u0 = 0 is
# [0.5, 0.5, 1.0, 1.0]
f_linear(u, p) = [u[1] + u[2] - 1.0, u[3] + u[4] - 2.0]
prob_linear = NonlinearLeastSquaresProblem(NonlinearFunction(f_linear), zeros(4))

sol = solve(prob_linear, LevenbergMarquardt(); maxiters = 1000, abstol = 1.0e-10)
@test SciMLBase.successful_retcode(sol)
@test norm(sol.resid) < 1.0e-8
@test sol.u ≈ [0.5, 0.5, 1.0, 1.0] atol = 1.0e-6

# The minimum-norm formulation must actually be engaged for underdetermined problems,
# and its damped JJᵀ system is SPD so the default linear solver must be Cholesky
cache = SciMLBase.init(prob_linear, LevenbergMarquardt(); abstol = 1.0e-10)
@test cache.descent_cache.descent_cache.mode isa Val{:minimum_norm}
@test cache.descent_cache.descent_cache.lincache.lincache.alg isa CholeskyFactorization

# ... and opting out via `min_norm_mode = :disabled` must fall back to the least squares
# formulation and still converge
descent = NonlinearSolveBase.DampedNewtonDescent(;
    initial_damping = 1.0,
    damping_fn = NonlinearSolveFirstOrder.LevenbergMarquardtDampingFunction(2.0, 3.0, 1.0e-8),
    min_norm_mode = :disabled
)
alg_disabled = GeneralizedFirstOrderAlgorithm(;
    descent, trustregion = NonlinearSolveFirstOrder.LevenbergMarquardtTrustRegion(1.0),
    name = :LevenbergMarquardt, concrete_jac = Val(true)
)
cache_disabled = SciMLBase.init(prob_linear, alg_disabled; abstol = 1.0e-10)
@test cache_disabled.descent_cache.mode isa Val{:least_squares}
sol_disabled = solve(prob_linear, alg_disabled; maxiters = 1000, abstol = 1.0e-10)
@test SciMLBase.successful_retcode(sol_disabled)
@test norm(sol_disabled.resid) < 1.0e-8

# Nonlinear underdetermined: 2 equations, 5 unknowns
f_nonlinear(u, p) = [sin(u[1]) + u[2]^2 + u[3] - 1.0, u[3] * u[4] + u[5] - 2.0]
prob_nonlinear = NonlinearLeastSquaresProblem(
    NonlinearFunction(f_nonlinear), [0.5, 0.5, 0.5, 2.0, 1.0]
)
sol = solve(prob_nonlinear, LevenbergMarquardt(); maxiters = 1000, abstol = 1.0e-8)
@test SciMLBase.successful_retcode(sol)
@test norm(sol.resid) < 1.0e-6

# In-place underdetermined: 2 equations, 4 unknowns
function f_iip!(resid, u, p)
    resid[1] = u[1] + u[2] + u[3] - 3.0
    resid[2] = u[1] * u[2] - u[4] - 1.0
    return nothing
end
prob_iip = NonlinearLeastSquaresProblem(
    NonlinearFunction{true}(f_iip!; resid_prototype = zeros(2)), ones(4)
)
sol = solve(prob_iip, LevenbergMarquardt(); maxiters = 1000, abstol = 1.0e-8)
@test SciMLBase.successful_retcode(sol)
@test norm(sol.resid) < 1.0e-6

# Badly scaled underdetermined system exercises the damping update on the JJᵀ system
f_scaled(u, p) = [1.0e4 * u[1] + u[2] - 1.0, u[3] + u[4] - 2.0]
prob_scaled = NonlinearLeastSquaresProblem(NonlinearFunction(f_scaled), zeros(4))
sol = solve(prob_scaled, LevenbergMarquardt(); maxiters = 1000, abstol = 1.0e-10)
@test SciMLBase.successful_retcode(sol)
@test norm(sol.resid) < 1.0e-8

# Linear wide-matrix check (suggested by @stevengj in #859): for f(x) = Ax - b starting
# at x = 0, the converged solution must match the minimum-norm solution A \ b, since every
# minimum-norm LM step Jᵀz lies in the row space of A.
rng = StableRNG(0)
for (m_lin, n_lin) in ((5, 20), (10, 40))
    A = randn(rng, m_lin, n_lin)
    b = randn(rng, m_lin)
    prob_lin = NonlinearLeastSquaresProblem(
        NonlinearFunction{false}((u, p) -> A * u - b), zeros(n_lin)
    )
    sol_lin = solve(prob_lin, LevenbergMarquardt(); maxiters = 1000, abstol = 1.0e-12)
    @test SciMLBase.successful_retcode(sol_lin)
    @test sol_lin.u ≈ A \ b atol = 1.0e-8
end

# Rank-deficient underdetermined system (duplicated row, consistent rhs): the damped JJᵀ
# system stays solvable and the minimum-norm solution is still recovered from x = 0
A_rd = [1.0 2.0 3.0 4.0; 1.0 2.0 3.0 4.0; 0.0 1.0 0.0 1.0]
b_rd = A_rd * [1.0, -1.0, 2.0, 0.5]
prob_rd = NonlinearLeastSquaresProblem(
    NonlinearFunction{false}((u, p) -> A_rd * u - b_rd), zeros(4)
)
sol = solve(prob_rd, LevenbergMarquardt(); maxiters = 1000, abstol = 1.0e-10)
@test SciMLBase.successful_retcode(sol)
@test norm(sol.resid) < 1.0e-8
@test sol.u ≈ pinv(A_rd) * b_rd atol = 1.0e-6

# The dual system JJᵀ + λD̃ᵀD̃ is symmetric positive definite, so Cholesky works
sol = solve(
    prob_linear, LevenbergMarquardt(; linsolve = CholeskyFactorization());
    maxiters = 1000, abstol = 1.0e-10
)
@test SciMLBase.successful_retcode(sol)
@test sol.u ≈ [0.5, 0.5, 1.0, 1.0] atol = 1.0e-6

# Underdetermined with StaticArrays (immutable path)
f_sa(u, p) = SA[u[1] + u[2] - 1.0, u[3] + u[4] - 2.0]
prob_sa = NonlinearLeastSquaresProblem(NonlinearFunction(f_sa), SA[0.0, 0.0, 0.0, 0.0])
sol = solve(prob_sa, LevenbergMarquardt(); maxiters = 1000, abstol = 1.0e-10)
@test SciMLBase.successful_retcode(sol)
@test norm(sol.resid) < 1.0e-8
@test sol.u ≈ [0.5, 0.5, 1.0, 1.0] atol = 1.0e-6

# Large underdetermined: 10 equations, 50 unknowns, solved through the small m×m system
function f_large(u, p)
    resid = zeros(eltype(u), 10)
    for i in 1:10
        start_idx = 5 * (i - 1) + 1
        resid[i] = sum(u[start_idx:(start_idx + 4)]) - Float64(i)
    end
    return resid
end
prob_large = NonlinearLeastSquaresProblem(NonlinearFunction(f_large), zeros(50))
sol = solve(prob_large, LevenbergMarquardt(); maxiters = 1000, abstol = 1.0e-8)
@test SciMLBase.successful_retcode(sol)
@test norm(sol.resid) < 1.0e-6
