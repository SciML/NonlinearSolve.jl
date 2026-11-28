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

@testset "Simple Newton residual at the accepted iterate" begin
    f(x, p) = 0.01f0 * (x^2 - 1)
    jac(x, p) = 0.02f0 * x
    x0 = 0.01f0
    for (problem, alg) in (
            (NonlinearProblem, SimpleNewtonRaphson()),
            (NonlinearLeastSquaresProblem, SimpleGaussNewton()),
        )
        scaled(x, tol) = solve(problem(NonlinearFunction(f; jac), x), alg; abstol = tol)
        for tol in (0.01f0, 1.0f-5)
            for sol in (scaled(x0, tol), Reactant.@jit(scaled(Reactant.ConcreteRNumber(x0), tol)))
                x = Float32(sol.u)
                @test sol.retcode == ReturnCode.Success
                @test Float32(sol.resid) ≈ f(x, nothing) atol = eps(Float32)
                @test abs(f(x, nothing)) ≤ tol
                if tol == 0.01f0
                    @test x == x0
                end
            end
        end
    end
end

@testset "Simple Newton evaluates the Jacobian once per attempted step" begin
    f(x, calls) = x .^ 2 .- 1
    function jac(x, calls)
        calls .+= 1
        return diagm(2 .* x)
    end
    @testset for (problem, alg) in (
            (NonlinearProblem, SimpleNewtonRaphson()),
            (NonlinearLeastSquaresProblem, SimpleGaussNewton()),
        )
        @testset for budget in (1, 0), compiled in (false, true)
            run(x, calls) = solve(
                problem(NonlinearFunction(f; jac), x, calls), alg;
                abstol = 0.01f0, maxiters = budget
            )
            executable = if compiled
                Reactant.@compile run(Reactant.to_rarray([2.0f0]), Reactant.to_rarray(zeros(Int32, 1)))
            else
                run
            end
            cases = budget == 0 ? ((2.0f0, 2.0f0, 0, ReturnCode.MaxIters),) :
                ((1.001f0, 1.001f0, 0, ReturnCode.Success), (2.0f0, 1.25f0, 1, ReturnCode.MaxIters))
            for (initial, expected, count, code) in cases
                x, calls = [initial], zeros(Int32, 1)
                if compiled
                    x, calls = Reactant.to_rarray(x), Reactant.to_rarray(calls)
                end
                sol = executable(x, calls)
                @test only(Array(calls)) == count
                @test sol.retcode == code
                @test Array(sol.u) == [expected]
                @test Array(sol.resid) ≈ f([expected], nothing) atol = eps(Float32)
                if count == 1
                    @test Array(sol.resid) == [0.5625f0]
                end
            end
        end
    end
end

@testset "Simple Broyden real and complex steps" begin
    f(x, p) = x ./ 2 .- p
    for alg in (SimpleBroyden(; alpha = 1.0f0), SimpleLimitedMemoryBroyden(; alpha = 1.0f0))
        run(u, p) = solve(NonlinearProblem(f, u, p), alg; maxiters = 2)
        for (u, p, expected) in (
                (Float32[0], Float32[2], Float32[4]),
                (ComplexF32[0, 0], ComplexF32[2im, 0], ComplexF32[4im, 0]),
            )
            @testset "$(typeof(alg)) $(eltype(u))" begin
                for compiled in (false, true)
                    sol = compiled ? Reactant.@jit(run(Reactant.to_rarray(u), Reactant.to_rarray(p))) : run(u, p)
                    actual = Array(sol.u)
                    @test actual == expected
                    @test iszero(f(actual, p))
                    @test iszero(Array(sol.resid))
                    @test sol.retcode == ReturnCode.Success
                end
            end
        end
    end
end
