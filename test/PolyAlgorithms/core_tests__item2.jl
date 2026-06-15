using NonlinearSolve

f(u, p) = u .* u .- 2
u0 = [1.0, 1.0]

prob = NonlinearProblem(f, u0)

polyalgs = (
    RobustMultiNewton(), FastShortcutNonlinearPolyalg(), nothing, missing,
    NonlinearSolvePolyAlgorithm((Broyden(), LimitedMemoryBroyden())),
)

@testset "Direct Solve" begin
    @testset for alg in polyalgs
        alg = alg === missing ? () : (alg,)
        sol = solve(prob, alg...; abstol = 1.0e-9)
        @test SciMLBase.successful_retcode(sol)
        err = maximum(abs, f(sol.u, 2.0))
        @test err < 1.0e-9
    end
end

@testset "Caching Interface" begin
    @testset for alg in polyalgs
        alg = alg === missing ? () : (alg,)
        cache = init(prob, alg...; abstol = 1.0e-9)
        solver = solve!(cache)
        @test SciMLBase.successful_retcode(solver)
        SciMLBase.reinit!(cache, u0)
    end
end

@testset "Step Interface" begin
    @testset for alg in polyalgs
        alg = alg === missing ? () : (alg,)
        cache = init(prob, alg...; abstol = 1.0e-9)
        for i in 1:10000
            step!(cache)
            cache.force_stop && break
        end
        @test SciMLBase.successful_retcode(cache.retcode)
    end
end
