using NonlinearSolveQuasiNewton, NonlinearSolveBase, LinearSolve, ForwardDiff, SciMLBase, Test

f(u, p) = [u[1]^2 - p, u[1] * u[2] - 1.0]
prob = NonlinearProblem(f, [1.0, 1.0], 2.0)
root(p) = [sqrt(p), inv(sqrt(p))]

# the linear-solve cache that inverts the Jacobian when Broyden (re)builds its inverse
inverse_solver(cache) = cache.linsolve_workspace.lincache.lincache.alg

default_alg = Broyden(; init_jacobian = Val(:true_jacobian))
alg = Broyden(; init_jacobian = Val(:true_jacobian), linsolve = LUFactorization())

@testset "linsolve selects the inverse-Jacobian solver" begin
    @test inverse_solver(init(prob, alg)) isa LUFactorization
    @test !(inverse_solver(init(prob, default_alg)) isa LUFactorization)

    sol = solve(prob, alg; abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ root(2.0) atol = 1.0e-8
    @test maximum(abs, f(sol.u, 2.0)) < 1.0e-8
end

@testset "linsolve is used again when the inverse is rebuilt" begin
    cache = init(prob, alg; abstol = 1.0e-10)
    solve!(cache)
    reinit!(cache; u0 = [1.0, 1.0], p = 3.0)
    SciMLBase.step!(cache)
    SciMLBase.step!(cache; recompute_jacobian = true)
    @test inverse_solver(cache) isa LUFactorization
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ root(3.0) atol = 1.0e-8
end

@testset "a descent's linsolve does not select the inverse-Jacobian solver" begin
    # an inverse-storing QuasiNewtonAlgorithm whose descent carries a linear solver keeps
    # LinearSolve's default algorithm for the inverse, as Broyden without `linsolve` does
    legacy = QuasiNewtonAlgorithm(;
        default_alg.linesearch, default_alg.update_rule, default_alg.reinit_rule,
        default_alg.initialization, default_alg.max_resets, default_alg.concrete_jac,
        default_alg.name,
        descent = NonlinearSolveBase.NewtonDescent(; linsolve = LUFactorization())
    )
    @test typeof(inverse_solver(init(prob, legacy))) ==
        typeof(inverse_solver(init(prob, default_alg)))
    @test !(inverse_solver(init(prob, legacy)) isa LUFactorization)
    sol = solve(prob, legacy; abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ root(2.0) atol = 1.0e-8
end
