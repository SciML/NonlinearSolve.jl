@testitem "Issue #451" tags=[:misc] begin
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
