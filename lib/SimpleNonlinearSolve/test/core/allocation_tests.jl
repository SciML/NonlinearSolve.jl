@itesitem "Allocation Tests" tags = [:alloc_check] begin
    using SimpleNonlinearSolve, StaticArrays, AllocCheck

    quadratic_f(u, p) = u .* u .- p
    quadratic_f!(du, u, p) = (du .= u .* u .- p)

    @testset "$(nameof(typeof(alg)))" for alg in (
            SimpleNewtonRaphson(),
            SimpleTrustRegion(),
            SimpleTrustRegion(; nlsolve_update_rule = Val(true)),
            SimpleBroyden(),
            SimpleLimitedMemoryBroyden(),
            SimpleKlement(),
            SimpleHalley(),
            SimpleBroyden(; linesearch = Val(true)),
            SimpleLimitedMemoryBroyden(; linesearch = Val(true)),
        )
        @check_allocs nlsolve(prob, alg) = SciMLBase.solve(prob, alg; abstol = 1.0e-9)

        nlprob_scalar = NonlinearProblem{false}(quadratic_f, 1.0, 2.0)
        nlprob_sa = NonlinearProblem{false}(quadratic_f, @SVector[1.0, 1.0], 2.0)

        try
            nlsolve(nlprob_scalar, alg)
            @test true
        catch e
            @error e
            @test false
        end

        # ForwardDiff allocates for hessian since we don't propagate the chunksize
        try
            nlsolve(nlprob_sa, alg)
            @test true
        catch e
            @error e
            @test false broken = (alg isa SimpleHalley)
        end
    end
end
