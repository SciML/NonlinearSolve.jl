@testitem "No warning tests" tags=[:misc] begin
    using NonlinearSolve

    f(u, p) = u .* u .- p
    u0 = [1.0, 1.0]
    p = 2.0
    prob = NonlinearProblem(f, u0, p)
    @test_nowarn solve(
        prob, NewtonRaphson(autodiff = AutoFiniteDiff(), linesearch = LineSearchesJL()))
end
