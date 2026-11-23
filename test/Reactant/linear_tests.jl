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
