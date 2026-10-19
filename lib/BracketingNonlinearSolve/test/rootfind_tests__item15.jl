using BracketingNonlinearSolve
using ADTypes: AutoForwardDiff
using ForwardDiff

@testset "ModABNewton finite brackets across scales" begin
    @testset "$T, $bounds, sign=$s, reverse=$reversed, mode=$mode" for T in (Float32, Float64),
            (bounds, root) in (
                ((T(1.0e0), T(1.7e0)), T(1.5e0)),
                ((floatmax(T) * T(0.6), floatmax(T) * T(0.95)), floatmax(T) * T(0.8)),
                ((-floatmax(T) * T(0.95), -floatmax(T) * T(0.6)), -floatmax(T) * T(0.8)),
                ((-floatmax(T), floatmax(T)), floatmax(T) / 4),
                ((floatmin(T), 4floatmin(T)), 3floatmin(T)),
                ((eps(zero(T)), 8eps(zero(T))), 5eps(zero(T))),
            ), s in (-one(T), one(T)), reversed in (false, true), mode in (:jac, :ad, :zero_jac)
        inside = Ref(true)
        f = function (x, p)
            inside[] &= isfinite(x) && bounds[1] <= x <= bounds[2]
            return bounds[1] < 0 < bounds[2] ? s * (x / 2 - root / 2) : s * (x - root)
        end
        df = (x, p) -> mode === :zero_jac ? zero(x) : (bounds[1] < 0 < bounds[2] ? s / 2 : s)
        fun = mode === :ad ? IntervalNonlinearFunction(f) : IntervalNonlinearFunction(f; jac = df)
        alg = mode === :ad ? ModABNewton(; autodiff = AutoForwardDiff()) : ModABNewton()
        prob = IntervalNonlinearProblem(fun, reversed ? reverse(bounds) : bounds)
        sol = solve(prob, alg; abstol = zero(T))
        @test inside[]
        @test SciMLBase.successful_retcode(sol)
        @test bounds[1] <= sol.left <= root <= sol.right <= bounds[2]
        @test sol.u == root
        @test iszero(f(sol.u, nothing))
    end

    @testset "Large decimal endpoints, mode=$mode" for mode in (:jac, :ad)
        f = (x, p) -> x - 1.5e308
        fun = mode === :jac ? IntervalNonlinearFunction(f; jac = (x, p) -> 1.0) : IntervalNonlinearFunction(f)
        alg = mode === :jac ? ModABNewton() : ModABNewton(; autodiff = AutoForwardDiff())
        sol = solve(IntervalNonlinearProblem(fun, (1.0e308, 1.7e308)), alg; abstol = 0.0)
        @test sol.retcode == ReturnCode.Success
        @test sol.u == 1.5e308
        @test sol.left == sol.right == 1.5e308
    end
end
