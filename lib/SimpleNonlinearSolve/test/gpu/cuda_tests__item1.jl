using SimpleNonlinearSolve

using StaticArrays, CUDA, SimpleNonlinearSolve, ADTypes, LineSearch

if CUDA.functional()
    CUDA.allowscalar(false)

    f(u, p) = u .* u .- 2
    f!(du, u, p) = (du .= u .* u .- 2)

    @testset "$(nameof(typeof(alg)))" for alg in (
            SimpleNewtonRaphson(; autodiff = AutoForwardDiff()),
            SimpleDFSane(),
            SimpleTrustRegion(; autodiff = AutoForwardDiff()),
            SimpleTrustRegion(;
                nlsolve_update_rule = Val(true), autodiff = AutoForwardDiff()
            ),
            SimpleBroyden(),
            SimpleLimitedMemoryBroyden(),
            SimpleKlement(),
            SimpleHalley(; autodiff = AutoForwardDiff()),
            SimpleBroyden(; linesearch = LiFukushimaLineSearch(; nan_maxiters = nothing)),
            SimpleLimitedMemoryBroyden(;
                linesearch = LiFukushimaLineSearch(; nan_maxiters = nothing)
            ),
        )
        # Static Arrays
        u0 = @SVector[1.0f0, 1.0f0]
        probN = NonlinearProblem{false}(f, u0)
        sol = solve(probN, alg; abstol = 1.0f-6)
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs, sol.resid) ≤ 1.0f-6

        # Regular Arrays
        u0 = [1.0, 1.0]
        probN = NonlinearProblem{false}(f, u0)
        sol = solve(probN, alg; abstol = 1.0f-6)
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs, sol.resid) ≤ 1.0f-6

        # Regular Arrays Inplace
        if !(alg isa SimpleHalley)
            u0 = [1.0, 1.0]
            probN = NonlinearProblem{true}(f!, u0)
            sol = solve(probN, alg; abstol = 1.0f-6)
            @test SciMLBase.successful_retcode(sol)
            @test maximum(abs, sol.resid) ≤ 1.0f-6
        end
    end
end
