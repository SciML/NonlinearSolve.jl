@testitem "Halley method" begin
    f(u, p) = u .* u .- p
    f!(fu, u, p) = fu .= u .* u .- p
    u0 = [1.0, 1.0]
    p = 2.0

    # out-of-place
    prob1 = NonlinearProblem(f, u0, p)
    sol1 = solve(prob1, Halley())
    @test sol1.u ≈ [sqrt(2.0), sqrt(2.0)]

    # in-place
    prob2 = NonlinearProblem(f!, u0, p)
    sol2 = solve(prob2, Halley())
    @test sol2.u ≈ [sqrt(2.0), sqrt(2.0)]
end
