using NonlinearSolve

f_oop(u, p) = u .* u .- 2
f_iip(du, u, p) = (du .= u .* u .- 2)

u0 = [1.0, 1.0]
prob_oop = NonlinearProblem(f_oop, u0)
prob_iip = NonlinearProblem(f_iip, u0)

polyalgs = (
    FastShortcutNonlinearPolyalg(),
    RobustMultiNewton(),
    NonlinearSolvePolyAlgorithm((NewtonRaphson(), Broyden())),
)

@testset "solve() inference" begin
    # Default algorithm (polyalgorithm)
    @test_nowarn @inferred solve(prob_oop)
    @test_nowarn @inferred solve(prob_iip)

    @testset for alg in polyalgs
        @test_nowarn @inferred solve(prob_oop, alg)
        @test_nowarn @inferred solve(prob_iip, alg)
    end
end

@testset "solve!() inference" begin
    # Default algorithm (polyalgorithm)
    cache = init(prob_oop)
    @test_nowarn @inferred solve!(cache)

    cache = init(prob_iip)
    @test_nowarn @inferred solve!(cache)

    @testset for alg in polyalgs
        cache = init(prob_oop, alg)
        @test_nowarn @inferred solve!(cache)

        cache = init(prob_iip, alg)
        @test_nowarn @inferred solve!(cache)
    end
end
