using BracketingNonlinearSolve
include("setup_rootfindingtestsnippet.jl")

prob = IntervalNonlinearProblem(quadratic_f, (1.0, 20.0), 2.0)
ϵ = eps(Float64) # least possible tol for all methods

@testset for alg in (Bisection(), Falsi(), ITP(), Muller())
    @testset for abstol in [0.1, 0.01, 0.001, 0.0001, 1.0e-5, 1.0e-6]
        sol = solve(prob, alg; abstol)
        result_tol = abs(sol.u - sqrt(2))
        @test result_tol < abstol
        # test that the solution is not calculated upto max precision
        @test result_tol > ϵ
    end
end

@testset for alg in (Brent(), Ridder(), ModAB(), nothing)
    # These solvers converge rapidly so as we lower tolerance below 0.01, it
    # converges with max precision to the solution
    @testset for abstol in [0.1]
        sol = solve(prob, alg; abstol)
        result_tol = abs(sol.u - sqrt(2))
        @test result_tol < abstol
        # test that the solution is not calculated upto max precision
        @test result_tol > ϵ
    end
end
