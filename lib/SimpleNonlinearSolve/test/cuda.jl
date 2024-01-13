using SimpleNonlinearSolve, StaticArrays, CUDA, Test

CUDA.allowscalar(false)

f(u, p) = u .* u .- 2
f!(du, u, p) = du .= u .* u .- 2

@testset "Solving on GPUs" begin
    for alg in (SimpleNewtonRaphson(), SimpleDFSane(), SimpleTrustRegion(), SimpleBroyden(),
        SimpleLimitedMemoryBroyden(), SimpleKlement(), SimpleHalley())
        @info "Testing $alg on CUDA"

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
        alg isa SimpleHalley && continue
        u0 = [1.0, 1.0]
        probN = NonlinearProblem{true}(f!, u0)
        sol = solve(probN, alg; abstol = 1.0f-6)
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs, sol.resid) ≤ 1.0f-6
    end
end

function kernel_function(prob, alg)
    solve(prob, alg; abstol = 1.0f-6, reltol = 1.0f-6)
    return nothing
end

@testset "CUDA Kernel Launch Test" begin
    prob = NonlinearProblem{false}(f, @SVector[1.0f0, 1.0f0])

    for alg in (SimpleNewtonRaphson(), SimpleDFSane(), SimpleTrustRegion(), SimpleBroyden(),
        SimpleLimitedMemoryBroyden(), SimpleKlement(), SimpleHalley())
        @test begin
            try
                @cuda kernel_function(prob, alg)
                @info "Successfully launched kernel for $(alg)."
                true
            catch err
                @error "Kernel Launch failed for $(alg)."
                false
            end
        end broken=(alg isa SimpleHalley)
    end
end
