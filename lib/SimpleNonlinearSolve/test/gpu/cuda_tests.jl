@testitem "Solving on GPUs" tags=[:cuda] skip=:(using CUDA; !CUDA.functional()) begin
    using StaticArrays, CUDA

    CUDA.allowscalar(false)

    f(u, p) = u .* u .- 2
    f!(du, u, p) = du .= u .* u .- 2

    @testset "$(nameof(typeof(alg)))" for alg in (
        SimpleNewtonRaphson(), SimpleDFSane(), SimpleTrustRegion(),
        SimpleTrustRegion(; nlsolve_update_rule = Val(true)),
        SimpleBroyden(), SimpleLimitedMemoryBroyden(), SimpleKlement(),
        SimpleHalley(), SimpleBroyden(; linesearch = Val(true)),
        SimpleLimitedMemoryBroyden(; linesearch = Val(true)))
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

@testitem "CUDA Kernel Launch Test" tags=[:cuda] skip=:(using CUDA; !CUDA.functional()) timeout=3600 begin
    using StaticArrays, CUDA

    CUDA.allowscalar(false)

    f(u, p) = u .* u .- 2
    f!(du, u, p) = du .= u .* u .- 2

    function kernel_function(prob, alg)
        solve(prob, alg)
        return nothing
    end

    prob = NonlinearProblem{false}(f, @SVector[1.0f0, 1.0f0])

    @testset "$(nameof(typeof(alg)))" for alg in (
        SimpleNewtonRaphson(), SimpleDFSane(), SimpleTrustRegion(),
        SimpleTrustRegion(; nlsolve_update_rule = Val(true)),
        SimpleBroyden(), SimpleLimitedMemoryBroyden(), SimpleKlement(),
        SimpleHalley(), SimpleBroyden(; linesearch = Val(true)),
        SimpleLimitedMemoryBroyden(; linesearch = Val(true)))
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
