using BracketingNonlinearSolve, ForwardDiff

@testset "Integer brackets" begin
    @testset "$constructor, bounds=$bounds, abstol=$tol" for constructor in (NewtonBisection,),
            bounds in ((1, 20), (20, 1), (-20, -1), (-1, -20)), tol in (nothing, 0.0)
        fun = IntervalNonlinearFunction((x, p) -> x * x - p; jac = (x, p) -> 2x)
        sol = solve(IntervalNonlinearProblem(fun, bounds, 2), constructor(); abstol = tol)
        lo, hi = minmax(sol.left, sol.right)
        reference = sign(first(bounds)) * sqrt(big(2))
        @test SciMLBase.successful_retcode(sol)
        @test lo <= reference <= hi
        @test min(bounds...) <= lo <= sol.u <= hi <= max(bounds...)
        @test sol.u isa AbstractFloat
    end
end
