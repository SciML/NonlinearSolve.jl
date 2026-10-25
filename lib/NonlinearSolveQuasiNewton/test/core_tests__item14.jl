using NonlinearSolveQuasiNewton, NonlinearSolveBase, LinearSolve, SciMLBase

f(u, p) = [u[1]^2 - p, u[1] * u[2] - 1.0]
prob = NonlinearProblem(f, [1.0, 1.0], 2.0)

# the linear-solve cache that inverts the Jacobian when Broyden (re)builds its inverse
inverse_solver(cache) = cache.linsolve_workspace.lincache.lincache.alg

alg = Broyden(; init_jacobian = Val(:true_jacobian), linsolve = LUFactorization())
@test inverse_solver(init(prob, alg)) isa LUFactorization
@test !(inverse_solver(init(prob, Broyden(; init_jacobian = Val(:true_jacobian)))) isa LUFactorization)

ref = solve(prob, Broyden(; init_jacobian = Val(:true_jacobian)); abstol = 1.0e-10)
sol = solve(prob, alg; abstol = 1.0e-10)
@test SciMLBase.successful_retcode(sol)
@test sol.u ≈ ref.u atol = 1.0e-8
