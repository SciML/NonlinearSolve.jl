using BracketingNonlinearSolve
include("setup_rootfindingtestsnippet.jl")

@testset for alg in (
        Alefeld(), Bisection(), Brent(), Falsi(), ITP(), Muller(), Ridder(), ModAB(), nothing,
    )
    f1(u, p) = u * u - p
    f2(u, p) = p - u * u

    for p in 1:4
        sol1 = solve(IntervalNonlinearProblem(f1, (1.0, 2.0), p), alg)
        sol2 = solve(IntervalNonlinearProblem(f2, (1.0, 2.0), p), alg)
        sol3 = solve(IntervalNonlinearProblem(f1, (2.0, 1.0), p), alg)
        sol4 = solve(IntervalNonlinearProblem(f2, (2.0, 1.0), p), alg)
        @test abs.(sol1.u) ≈ sqrt.(p)
        @test abs.(sol2.u) ≈ sqrt.(p)
        @test abs.(sol3.u) ≈ sqrt.(p)
        @test abs.(sol4.u) ≈ sqrt.(p)
        # Test brackets consistency
        @test sol1.left ≤ sol1.right
        @test sol2.left ≤ sol2.right
        @test sol3.left ≥ sol3.right
        @test sol4.left ≥ sol4.right
    end
end
