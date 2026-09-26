using BracketingNonlinearSolve
include("setup_rootfindingtestsnippet.jl")

prob = IntervalNonlinearProblem(quadratic_f, (1.0, 20.0), 2.0)
ϵ = eps(Float64) # least possible tol for all methods

@testset for alg in (Alefeld(), Bisection(), Falsi(), ITP(), Muller())
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

@testset "Alefeld honors abstol (#1315)" begin
    abstol_loose = 1.0e-3
    abstol_tight = 1.0e-12
    nfe_loose = Ref(0)
    nfe_tight = Ref(0)
    f_loose = (u, p) -> (nfe_loose[] += 1; u * u - p)
    f_tight = (u, p) -> (nfe_tight[] += 1; u * u - p)

    sol_loose = solve(
        IntervalNonlinearProblem(f_loose, (1.0, 20.0), 2.0), Alefeld();
        abstol = abstol_loose
    )
    sol_tight = solve(
        IntervalNonlinearProblem(f_tight, (1.0, 20.0), 2.0), Alefeld();
        abstol = abstol_tight
    )

    half_loose = abs(sol_loose.right - sol_loose.left) / 2
    @test sol_loose.retcode == ReturnCode.Success
    @test half_loose < abstol_loose
    @test sol_loose.left ≤ sqrt(2) ≤ sol_loose.right
    @test nfe_loose[] < nfe_tight[]
end
