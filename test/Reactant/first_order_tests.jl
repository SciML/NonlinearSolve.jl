@testset "Rejected trust-region trial leaves the iterate and residual intact" begin
    f(u, p) = u .^ 2 .- 1
    jac(u, p) = diagm(2 .* u)
    correct(u, old, p, cache) = isnothing(cache) ? u : one.(u)
    function rejected_trial(u)
        return solve(
            NonlinearProblem(NonlinearFunction(f; jac), u),
            TrustRegion(; initial_trust_radius = 100.0f0);
            postcondition = correct, maxiters = 1,
            termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs))
        )
    end
    u = fill(0.1f0, 2)
    for sol in (rejected_trial(u), Reactant.@jit(rejected_trial(Reactant.to_rarray(u))))
        @test sol.retcode == ReturnCode.MaxIters
        @test Array(sol.u) == u
        @test Array(sol.resid) ≈ f(u, nothing)
    end
end

@testset "Compilation reports unsupported Jacobian reuse" begin
    f(u, p) = u .^ 2 .- 2
    jac(u, p) = diagm(2 .* u)
    function modified_newton(u)
        return solve(
            NonlinearProblem(NonlinearFunction(f; jac), u),
            NewtonRaphson(; jacobian_reuse = JacobianReuse(; max_age = 10, max_residual_ratio = 1));
            abstol = 0.01f0, maxiters = 2,
            termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs))
        )
    end
    u = ones(Float32, 2)
    sol = modified_newton(u)
    @test sol.retcode == ReturnCode.MaxIters
    @test sol.u == fill(1.375f0, 2)
    @test_throws ArgumentError Reactant.@compile modified_newton(Reactant.to_rarray(u))
end
