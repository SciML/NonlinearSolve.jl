@testset "Singular explicit LU under compilation" begin
    f(u, p) = one.(u)
    jac(u, p) = zero.(u .* u')
    nf = NonlinearFunction(f; jac)
    function singular_newton(u)
        return solve(
            NonlinearProblem(nf, u), NewtonRaphson(; linsolve = LUFactorization());
            maxiters = 2, termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs))
        )
    end
    u = ones(Float32, 2)
    @test singular_newton(u).retcode == ReturnCode.InternalLinearSolveFailed
    @test_throws ArgumentError Reactant.@compile singular_newton(Reactant.to_rarray(u))
end

@testset "Custom matrix solve options require host execution" begin
    f(u, p) = u .^ 2 .- 2
    jac(u, p) = diagm(0 => 2 .* u)
    function solve_with_options(u)
        return solve(
            NonlinearProblem(NonlinearFunction(f; jac), u), NewtonRaphson();
            linsolve_kwargs = (; maxiters = 10)
        )
    end
    u = ones(Float32, 2)
    @test solve_with_options(u).u ≈ fill(sqrt(2.0f0), 2)
    @test_throws "Run the solve outside Reactant compilation" Reactant.@compile solve_with_options(Reactant.to_rarray(u))
end

@testset "Dense Newton retains small nonzero pivots" begin
    f(u, A) = A * u .- 1
    jac(u, A) = A
    function scaled_newton(u, A)
        return solve(
            NonlinearProblem(NonlinearFunction(f; jac), u, A), NewtonRaphson();
            abstol = 1.0f-6, maxiters = 8
        )
    end
    u, A = zeros(Float32, 2), diagm(Float32[1, 2.0^-26])
    compiled = Reactant.@jit scaled_newton(Reactant.to_rarray(u), Reactant.to_rarray(A))
    for sol in (scaled_newton(u, A), compiled)
        @test Array(sol.u) == Float32[1, 2^26]
        @test f(Array(sol.u), A) == Array(sol.resid) == zeros(Float32, 2)
        @test sol.retcode == ReturnCode.Success
    end
    inverse(A) = NonlinearSolveBase.Utils.linsolve_identity!!(nothing, A)
    for (pivot, expected) in ((2.0f0^-26, 2.0f0^26), (0.0f0, 0.0f0))
        result = Reactant.@jit inverse(Reactant.to_rarray(diagm(Float32[1, pivot])))
        @test Array(result) == diagm(Float32[1, expected])
    end
end
