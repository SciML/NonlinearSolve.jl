@testset "Safe-best Newton retains the first iterate with unit residual" begin
    f(u, p) = exp.(u) .+ 1.0f0
    jac(u, p) = diagm(0 => exp.(u))
    function run_safe_best(u)
        return solve(
            NonlinearProblem(NonlinearFunction(f; jac), u), NewtonRaphson(); maxiters = 8
        )
    end
    u0 = Float32[1]
    expected = only(u0)
    for _ in 1:3
        expected -= (exp(expected) + 1.0f0) / exp(expected)
    end
    @test exp(expected) + 1.0f0 == 1.0f0
    host = run_safe_best(u0)
    compiled = Reactant.@jit run_safe_best(Reactant.to_rarray(u0))
    @test host.retcode == compiled.retcode == ReturnCode.MaxIters
    @test only(host.u) ≈ expected
    @test only(Array(compiled.u)) ≈ expected
    @test Array(compiled.u) ≈ host.u
    @test Array(compiled.resid) == host.resid == Float32[1]
end

@testset "Plain absolute norm continues with nonfinite residuals" begin
    f(u, p) = u .- p
    jac(u, p) = diagm(0 => one.(u))
    function run_plain(u, p)
        return solve(
            NonlinearProblem(NonlinearFunction(f; jac), u, p), NewtonRaphson();
            termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs)), maxiters = 3
        )
    end
    u, p = Float32[1], Float32[NaN]
    host = run_plain(u, p)
    compiled = Reactant.@jit run_plain(Reactant.to_rarray(u), Reactant.to_rarray(p))
    @test host.retcode == compiled.retcode == ReturnCode.MaxIters
    @test all(isnan, host.resid)
    @test all(isnan, Array(compiled.resid))
end

@testset "Diagonal cubic Broyden exhausts plain-norm iterations" begin
    function run_cubic(u, p)
        return solve(
            NonlinearProblem((x, y) -> x .^ 3 .- y, u, p), Broyden(; update_rule = Val(:diagonal));
            maxiters = 50, abstol = 1.0f-5,
            termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs))
        )
    end
    u, p = Float32[3, -2], Float32[2]
    host = run_cubic(u, p)
    compiled = Reactant.@jit run_cubic(Reactant.to_rarray(u), Reactant.to_rarray(p))
    @test host.retcode == compiled.retcode == ReturnCode.MaxIters
    @test isequal(Array(compiled.u), host.u)
    @test isequal(Array(compiled.resid), host.resid)
end
