@testitem "PolyAlgorithm Aliasing" tags=[:misc] begin
    using NonlinearProblemLibrary

    # Use a problem that the initial solvers cannot solve and cause the initial value to
    # diverge. If we don't alias correctly, all the subsequent algorithms will also fail.
    prob = NonlinearProblemLibrary.nlprob_23_testcases["Generalized Rosenbrock function"].prob
    u0 = copy(prob.u0)
    prob = remake(prob; u0 = copy(u0))

    # If aliasing is not handled properly this will diverge
    sol = solve(prob; abstol = 1e-6, alias_u0 = true,
        termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs)))

    @test sol.u === prob.u0
    @test SciMLBase.successful_retcode(sol.retcode)

    prob = remake(prob; u0 = copy(u0))

    cache = init(prob; abstol = 1e-6, alias_u0 = true,
        termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs)))
    sol = solve!(cache)

    @test sol.u === prob.u0
    @test SciMLBase.successful_retcode(sol.retcode)
end
