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
