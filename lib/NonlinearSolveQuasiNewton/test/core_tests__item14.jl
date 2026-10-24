using NonlinearSolveQuasiNewton, NonlinearSolveBase, LinearSolve, SciMLBase

f(u, p) = [u[1]^2 - p, u[1] * u[2] - 1.0]
prob = NonlinearProblem(f, [1.0, 1.0], 2.0)

for init_jacobian in (Val(:identity), Val(:true_jacobian))
    alg = Broyden(; init_jacobian, linsolve = LUFactorization())
    @test NonlinearSolveBase.get_linear_solver(alg.descent) isa LUFactorization
    ref = solve(prob, Broyden(; init_jacobian); abstol = 1.0e-10)
    sol = solve(prob, alg; abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ ref.u atol = 1.0e-8
end
@test NonlinearSolveBase.get_linear_solver(Broyden().descent) === nothing
