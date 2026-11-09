using BracketingNonlinearSolve, ForwardDiff
using ADTypes: AutoForwardDiff

@testset "Integer brackets" begin
    @testset "$constructor, bounds=$bounds, abstol=$tol, mode=$mode" for constructor in (NewtonBisection,),
            bounds in ((1, 20), (20, 1), (-20, -1), (-1, -20)), tol in (nothing, 0.0), mode in (:jac, :ad)
        fun = mode === :jac ? IntervalNonlinearFunction((x, p) -> x * x - p; jac = (x, p) -> 2x) : IntervalNonlinearFunction((x, p) -> x * x - p)
        alg = mode === :jac ? constructor() : constructor(; autodiff = AutoForwardDiff())
        sol = solve(IntervalNonlinearProblem(fun, bounds, 2), alg; abstol = tol)
        lo, hi = minmax(sol.left, sol.right)
        reference = sign(first(bounds)) * sqrt(big(2))
        @test SciMLBase.successful_retcode(sol)
        @test lo <= reference <= hi
        @test min(bounds...) <= lo <= sol.u <= hi <= max(bounds...)
        @test sol.u isa AbstractFloat
    end
end
