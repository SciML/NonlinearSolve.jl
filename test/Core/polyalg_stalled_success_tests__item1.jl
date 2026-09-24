using NonlinearSolve
using Test

@testset "cache solve escalates after StalledSuccess" begin
    x = 1.0:10.0
    expected = [3.0, 2.0, 0.7]
    y = @. expected[1] + expected[2] * x^expected[3]
    residual(u, p) = @. y - (u[1] + u[2] * x^u[3])
    prob = NonlinearLeastSquaresProblem(
        residual, [0.5, 0.5, 0.5];
        lb = zeros(3), ub = [10.0, 10.0, 1.0]
    )

    direct_sol = solve(prob)
    cached_sol = solve!(init(prob))

    @test direct_sol.u ≈ expected atol = 1.0e-7
    @test cached_sol.u ≈ direct_sol.u atol = 1.0e-7
    @test cached_sol.resid ≈ direct_sol.resid atol = 1.0e-7
    @test cached_sol.retcode == direct_sol.retcode
end
