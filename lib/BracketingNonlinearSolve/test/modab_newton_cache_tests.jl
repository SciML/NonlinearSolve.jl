using BracketingNonlinearSolve

@testset "User Jacobian shares the residual cache" begin
    @testset "$T, bounds=$bounds" for T in (Float32, Float64, BigFloat),
            bounds in ((T(1), T(20)), (T(20), T(1)), (T(-20), T(-1)), (T(-1), T(-20)))
        last_x = Ref(T(NaN))
        cached_jac = Ref(zero(T))
        f = function (x, p)
            last_x[] = x
            cached_jac[] = 2x
            return x * x - p
        end
        jac = function (x, p)
            @test last_x[] == x
            return cached_jac[]
        end
        sol = solve(IntervalNonlinearProblem(IntervalNonlinearFunction(f; jac), bounds, T(2)), ModABNewton(); abstol = zero(T))
        lo, hi = minmax(sol.left, sol.right)
        reference = sign(first(bounds)) * sqrt(big(2))
        @test SciMLBase.successful_retcode(sol)
        @test lo <= reference <= hi
    end
end
