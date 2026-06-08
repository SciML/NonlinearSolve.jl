using NonlinearSolve

using SciMLBase

f(u, p) = u .- p
u0 = [5.0, 5.0]
p = [1.0, 2.0]
nf = NonlinearFunction(f; resid_prototype = zeros(2))
alg = LevenbergMarquardt()
@test !SciMLBase.allowsbounds(alg)

@testset "lower bound only" begin
    prob = NonlinearLeastSquaresProblem(nf, u0, p; lb = [-Inf, 0.0])
    sol = solve(prob, alg)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [1.0, 2.0] atol = 1.0e-6
end

@testset "upper bound only" begin
    prob = NonlinearLeastSquaresProblem(nf, u0, p; ub = [Inf, 10.0])
    sol = solve(prob, alg)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [1.0, 2.0] atol = 1.0e-6
end

@testset "lower bound excludes solution" begin
    prob = NonlinearLeastSquaresProblem(nf, u0, p; lb = [3.0, 3.0])
    sol = solve(prob, alg)
    @test all(sol.u .>= 3.0)
end
