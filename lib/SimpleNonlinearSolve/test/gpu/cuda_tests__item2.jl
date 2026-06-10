using SimpleNonlinearSolve

using StaticArrays, CUDA, SimpleNonlinearSolve, ADTypes, LineSearch
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
                SimpleBroyden(; linesearch = LiFukushimaLineSearch(; nan_maxiters = nothing)),
                SimpleLimitedMemoryBroyden(;
                    linesearch = LiFukushimaLineSearch(; nan_maxiters = nothing)
                ),
            )
            @test begin
                @cuda kernel_function(prob, alg)
                @info "Successfully launched kernel for $(alg)."
                true
            end
        end
    end
end
