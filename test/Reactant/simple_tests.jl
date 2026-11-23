using StaticArrays: SVector

@testset "SimpleHalley distinguishes singular and small pivots" begin
    function singular_halley(u, p)
        return solve(
            NonlinearProblem(nonlinear_function, u, p), SimpleHalley();
            abstol = 1.0f-4, maxiters = 1
        )
    end
    u = Float32[0, 1]
    ru = Reactant.to_rarray(u)
    compiled = Reactant.@compile singular_halley(ru, Reactant.to_rarray(Float32[1, 2]))
    for p in (Float32[1.0f-5, 1], Float32[1, 2])
        for sol in (singular_halley(u, p), compiled(ru, Reactant.to_rarray(p)))
            @test sol.retcode == ReturnCode.Unstable
            @test Array(sol.u) == u
            @test Array(sol.resid) == u .^ 2 .- p
        end
    end
    small = fill(1.0f-25, 2)
    for sol in (
            singular_halley(small, small),
            compiled(Reactant.to_rarray(small), Reactant.to_rarray(small)),
        )
        @test sol.retcode == ReturnCode.Success
        @test Array(sol.u) == small
    end
end

@testset "SimpleBroyden takes a full step without a line search" begin
    function unit_broyden(u, p)
        return solve(
            NonlinearProblem((x, y) -> x .- y, u, p), SimpleBroyden(; alpha = 1.0f0);
            maxiters = 1
        )
    end
    u, p = zeros(Float32, 2), Float32[2, -3]
    sol = Reactant.@jit unit_broyden(Reactant.to_rarray(u), Reactant.to_rarray(p))
    @test sol.retcode == ReturnCode.Success
    @test Array(sol.u) == p
    @test iszero(Array(sol.resid))
end

@testset "SimpleLimitedMemoryBroyden low-rank iteration" begin
    function low_rank_broyden(u, p)
        prob = NonlinearProblem((x, y) -> x .^ 2 .- y, u, p)
        return solve(prob, SimpleLimitedMemoryBroyden(; threshold = Val(2)))
    end
    u, p = ones(Float32, 4), Float32[2]
    host = low_rank_broyden(u, p)
    @test host.retcode == ReturnCode.Success
    @test host.u ≈ fill(sqrt(2.0f0), 4)
    @test_throws ArgumentError Reactant.@compile low_rank_broyden(
        Reactant.to_rarray(u), Reactant.to_rarray(p)
    )
    function static_broyden(p)
        initial = SVector(1.0f0, 1.0f0, 1.0f0, 1.0f0)
        return solve(
            NonlinearProblem((x, y) -> x .- y, initial, p),
            SimpleLimitedMemoryBroyden(; threshold = Val(2))
        )
    end
    static_host = static_broyden(2.0f0)
    @test static_host.retcode == ReturnCode.Success
    @test static_host.u == fill(2.0f0, 4)
    @test_throws ArgumentError Reactant.@compile static_broyden(Reactant.ConcreteRNumber(2.0f0))
end

@testset "SimpleKlement with a nonfinite residual" begin
    u, p = ones(Float32, 2), Float32[NaN]
    host = solve_one_step(u, p)
    sol = Reactant.@jit solve_one_step(Reactant.to_rarray(u), Reactant.to_rarray(p))
    @test host.retcode == sol.retcode == ReturnCode.MaxIters
    @test all(isnan, host.resid)
    @test all(isnan, Array(sol.resid))
end
