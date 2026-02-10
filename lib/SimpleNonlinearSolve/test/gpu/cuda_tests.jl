@testitem "Solving on CUDA" tags = [:cuda] begin
    using StaticArrays, CUDA, SimpleNonlinearSolve, ADTypes

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
                SimpleBroyden(; linesearch = Val(true)),
                SimpleLimitedMemoryBroyden(; linesearch = Val(true)),
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
end

@testitem "CUDA Kernel Launch Test" tags = [:cuda] begin
    using StaticArrays, CUDA, SimpleNonlinearSolve, ADTypes
    using NonlinearSolveBase: ImmutableNonlinearProblem

    if CUDA.functional()
        CUDA.allowscalar(false)

        f(u, p) = u .* u .- p

        function kernel_function(prob, alg)
            solve(prob, alg)
            return nothing
        end

        @testset for u0 in (1.0f0, @SVector[1.0f0, 1.0f0])
            prob = convert(ImmutableNonlinearProblem, NonlinearProblem{false}(f, u0, 2.0f0))

            # Note: SimpleHalley is excluded from kernel tests due to dynamic dispatch issues
            # (requires LU factorization, second derivatives, and complex type-dependent operations)
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
                    # SimpleHalley(; autodiff = AutoForwardDiff()),
                    SimpleBroyden(; linesearch = Val(true)),
                    SimpleLimitedMemoryBroyden(; linesearch = Val(true)),
                )
                @test begin
                    @cuda kernel_function(prob, alg)
                    @info "Successfully launched kernel for $(alg)."
                    true
                end
            end
        end
    end
end
