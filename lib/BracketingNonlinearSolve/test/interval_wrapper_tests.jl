using BracketingNonlinearSolve, ForwardDiff

quadratic(x, p) = x * x - 2
quadratic_jac(x, p) = 2x
solve_allocations(prob, alg) = @allocated solve(prob, alg; abstol = 0.0)

@testset "Interval wrapper inference and allocations" begin
    @testset "$alg, dual=$dual, jac=$supplied" for alg in (ModAB(), ITP(), Brent(), nothing),
            dual in (false, true), supplied in (false, true)
        f = supplied ? IntervalNonlinearFunction(quadratic; jac = quadratic_jac) : IntervalNonlinearFunction(quadratic)
        bounds = dual ? (ForwardDiff.Dual(1.0, 0.0), ForwardDiff.Dual(20.0, 0.0)) : (1.0, 20.0)
        prob = IntervalNonlinearProblem(f, bounds)
        sol = @inferred solve(prob, alg; abstol = 0.0)
        @test SciMLBase.successful_retcode(sol)
        @test sol.left <= sqrt(BigFloat(2)) <= sol.right
        @test solve_allocations(prob, alg) == 0
    end
end
