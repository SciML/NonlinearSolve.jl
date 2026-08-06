using BracketingNonlinearSolve
include("setup_rootfindingtestsnippet.jl")

ϵ = eps(Float64)

@testset "Problem kwargs forwarded to solvers (#522)" begin
    abstol = 1.0e-3
    # Default abstol when omitted is ~eps^(4/5) ≈ 3e-13. With problem abstol, solvers
    # that honor a loose tolerance must not refine to max precision (same check as
    # the tolerance tests). Solve kwargs must override problem kwargs.
    prob = IntervalNonlinearProblem(quadratic_f, (1.0, 20.0), 2.0; abstol)

    @testset for alg in (Bisection(), Falsi(), ITP(), Muller())
        sol_from_prob = solve(prob, alg)
        sol_from_solve = solve(
            IntervalNonlinearProblem(quadratic_f, (1.0, 20.0), 2.0), alg; abstol
        )

        result_tol = abs(sol_from_prob.u - sqrt(2))
        @test result_tol < abstol
        @test result_tol > ϵ
        @test sol_from_prob.u == sol_from_solve.u
        @test sol_from_prob.left == sol_from_solve.left
        @test sol_from_prob.right == sol_from_solve.right

        sol_override = solve(prob, alg; abstol = 1.0e-6)
        @test abs(sol_override.u - sqrt(2)) < 1.0e-6
    end

    # Brent / Ridder / ModAB can overshoot loose abstol to machine precision; still
    # require problem kwargs and solve kwargs to produce the same bracket.
    @testset for alg in (Brent(), Ridder(), ModAB())
        sol_from_prob = solve(prob, alg)
        sol_from_solve = solve(
            IntervalNonlinearProblem(quadratic_f, (1.0, 20.0), 2.0), alg; abstol
        )
        @test sol_from_prob.u == sol_from_solve.u
        @test sol_from_prob.left == sol_from_solve.left
        @test sol_from_prob.right == sol_from_solve.right
    end
end
