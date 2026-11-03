using BracketingNonlinearSolve

# Issue #860
a, b, c = 0.28299683548569865, 0.23110561490429102, -0.04293615418571393
f = (x, _) -> a * x^2 + b * x + c
tspan = (-18.26561304888476, -0.8216404439033063)

prob = IntervalNonlinearProblem{false}(f, tspan)
sol = solve(prob, ModAB(); abstol = 0.0, reltol = 0.0)

@test sol.retcode == ReturnCode.FloatingPointLimit
@test sol.left == -0.9726263117443217
@test sol.right == -0.9726263117443216

@testset "Anderson-Bjork halves the residual without shrinking the bracket" begin
    # Counterexample by L. Tomov: every Anderson-Bjork step lands left of the root and
    # halves the residual, so the old residual test never fell back to bisection. The
    # right end stayed at 1, and after 778 evaluations `-x * x` underflowed to an exact
    # zero near -1e-162. Residual-only steps are now limited, so the bracket shrinks.
    xs = Float64[]
    f = (x, _) -> (push!(xs, x); x < 0 ? -x * x : 7 * x * x)
    abstol = 1.0e-12
    sol = solve(IntervalNonlinearProblem(f, (-Float64(π), 1.0)), ModAB(); abstol)

    @test SciMLBase.successful_retcode(sol)
    @test sol.left < 0 < sol.right # A real bracket, not an underflowed exact zero
    @test sol.right - sol.left < 2abstol
    @test count(>(0), xs) > 1 # The right end of the bracket moves
    @test length(xs) ≤ 100
end
