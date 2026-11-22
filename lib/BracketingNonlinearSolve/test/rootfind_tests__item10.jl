using BracketingNonlinearSolve

# Record every point `f` is evaluated at.
function traced_solve(f, tspan, alg; abstol)
    xs = Float64[]
    prob = IntervalNonlinearProblem{false}((u, p) -> (push!(xs, u); f(u)), tspan)
    sol = solve(prob, alg; abstol)
    return sol, xs
end

@testset "Brent evaluates each point once" begin
    # The previous implementation re-evaluated `f(c)` at the top of every
    # iteration, where `c` is always a point whose value is already known: on
    # `x^2 - 2` over (1, 2) that was 20 evaluations at 9 distinct points.
    sol, xs = traced_solve(x -> x^2 - 2, (1.0, 2.0), Brent(); abstol = 1.0e-15)
    @test SciMLBase.successful_retcode(sol)
    @test abs(sol.u - sqrt(2)) ≤ 2 * eps(sqrt(2))
    @test length(unique(xs)) == length(xs)
    @test length(xs) ≤ 10
end

@testset "Brent takes the minimum step" begin
    # When an interpolated step would move the estimate by less than
    # `tol1 = 2 eps(b) + abstol`, Brent's method moves by `tol1` towards the
    # contrapoint instead. On a convex function every interpolated iterate lands on
    # one side of the root, and without that step the far end of the bracket is
    # replaced only by bisection. `x^4 - 2` over (0, 4) shows it: 12 evaluations
    # with the minimum step, 50 without it, and 62 (at 30 distinct points) for the
    # previous implementation.
    sol, xs = traced_solve(x -> x^4 - 2, (0.0, 4.0), Brent(); abstol = 5.0e-11)
    @test SciMLBase.successful_retcode(sol)
    @test abs(sol.u - 2^0.25) ≤ 1.0e-10
    @test sol.left ≤ 2^0.25 ≤ sol.right
    @test length(xs) ≤ 14
end
