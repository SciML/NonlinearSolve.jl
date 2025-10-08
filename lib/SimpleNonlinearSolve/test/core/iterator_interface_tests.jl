@testitem "Iterator Interface Error" tags=[:core] begin
    using SimpleNonlinearSolve

    # Test that Simple algorithms properly error when used with the iterator interface
    f(u, p) = u .* u .- 2.0
    u0 = 1.5
    prob = NonlinearProblem(f, u0)

    # Test with various Simple algorithms
    for alg in [SimpleNewtonRaphson(), SimpleBroyden(), SimpleTrustRegion(),
        SimpleDFSane(), SimpleKlement(), SimpleLimitedMemoryBroyden()]
        @test_throws ErrorException init(prob, alg)

        # Verify the error message contains helpful information
        try
            init(prob, alg)
            @test false  # Should not reach here
        catch e
            msg = sprint(showerror, e)
            @test occursin("iterator interface", msg)
            @test occursin("Simple algorithms", msg)
            @test occursin("NewtonRaphson()", msg)
            @test occursin("solve(prob, alg)", msg)
        end
    end

    # Verify that solve() still works correctly
    for alg in [SimpleNewtonRaphson(), SimpleBroyden(), SimpleTrustRegion()]
        sol = solve(prob, alg)
        @test sol.retcode == ReturnCode.Success
        @test abs(sol.u^2 - 2.0) < 1e-6
    end
end
