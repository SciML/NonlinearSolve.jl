@testitem "Scalar Jacobians: Issue #451" tags = [:core] begin
    f(u, p) = u^2 - p

    jac_calls = 0
    function df(u, p)
        global jac_calls += 1
        return 2u
    end

    fn = NonlinearFunction(f; jac = df)
    prob = NonlinearProblem(fn, 1.0, 2.0)
    sol = solve(prob, NewtonRaphson())
    @test sol.retcode == ReturnCode.Success
    @test jac_calls ≥ 1

    jac_calls = 0
    fn2 = NonlinearFunction(f)
    prob = NonlinearProblem(fn2, 1.0, 2.0)
    sol = solve(prob, NewtonRaphson())
    @test sol.retcode == ReturnCode.Success
    @test jac_calls == 0
end

@testitem "Dual of BigFloat: Issue #512" tags = [:core] begin
    using NonlinearSolveFirstOrder, ForwardDiff
    fn_iip = NonlinearFunction{true}((du, u, p) -> du .= u .* u .- p)
    u2 = [
        ForwardDiff.Dual(BigFloat(1.0), 5.0), ForwardDiff.Dual(BigFloat(1.0), 5.0),
        ForwardDiff.Dual(BigFloat(1.0), 5.0),
    ]
    prob_iip_bf = NonlinearProblem{true}(fn_iip, u2, ForwardDiff.Dual(BigFloat(2.0), 5.0))
    sol = solve(prob_iip_bf, NewtonRaphson())
    @test sol.retcode == ReturnCode.Success
end

@testitem "TrustRegion reinit! resets trust_region" tags = [:core] begin
    using NonlinearSolveFirstOrder, SciMLBase

    f(u, p) = u .* u .- p
    prob = NonlinearProblem(f, [1.0, 1.0], 2.0)
    cache = init(prob, TrustRegion())

    # Get the trust region cache
    tr_cache = cache.trustregion_cache
    @test tr_cache.trust_region == tr_cache.initial_trust_radius

    # Solve problem to modify the trust region
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    @test tr_cache.trust_region != tr_cache.initial_trust_radius

    # Reinitialize and check the trust region was reset
    reinit!(cache, [1.0, 1.0]; p = 2.0)
    @test tr_cache.trust_region == tr_cache.initial_trust_radius
end
