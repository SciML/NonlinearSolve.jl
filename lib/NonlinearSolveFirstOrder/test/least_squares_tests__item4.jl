using NonlinearSolveFirstOrder, SciMLBase, Test

function nan_residual!(du, u, p)
    du[1] = exp(u[1]) - exp(2 * u[1]) + 1.0
    du[2] = u[2] - 1.0
    return nothing
end

@testset "non-finite NLLS residual" begin
    @testset "$name" for (name, solver) in (
            ("TrustRegion", TrustRegion()),
            ("LevenbergMarquardt", LevenbergMarquardt()),
            ("GaussNewton", GaussNewton()),
        )
        prob = NonlinearLeastSquaresProblem(
            NonlinearFunction(nan_residual!; resid_prototype = zeros(2)), [1000.0, 0.0]
        )
        sol = solve(prob, solver; maxiters = 100)
        @test isnan(sol.resid[1])
        @test !SciMLBase.successful_retcode(sol)
    end
end
