using BracketingNonlinearSolve, ForwardDiff

@testset "Safeguarded Newton brackets across scales" begin
    @testset "$T, sign=$sgn, reverse=$rev, mode=$mode" for T in (Float32, Float64),
            sgn in (-one(T), one(T)), rev in (false, true), mode in (:jac, :zero_jac)
        a = ldexp(T(1.25), exponent(floatmax(T)))
        b = ldexp(T(1.75), exponent(floatmax(T)))
        r = T(3) / 8
        c = 3r^2 - 2r^3
        lo, hi = sgn > 0 ? (a, b) : (-b, -a)
        inside = Ref(true)
        f = function (x, p)
            inside[] &= isfinite(x) && lo <= ForwardDiff.value(x) <= hi
            s = (sgn * x - a) / (b - a)
            return 3s^2 - 2s^3 - c
        end
        df = function (x, p)
            s = (sgn * x - a) / (b - a)
            return mode === :zero_jac ? zero(x) : sgn * 6s * (1 - s) / (b - a)
        end
        reference = setprecision(256) do
            sgn * (BigFloat(a) + BigFloat(r) * (BigFloat(b) - BigFloat(a)))
        end
        fun = IntervalNonlinearFunction(f; jac = df)
        alg = NewtonBisection()
        sol = solve(IntervalNonlinearProblem(fun, rev ? (hi, lo) : (lo, hi)), alg; abstol = zero(T))
        left, right = minmax(sol.left, sol.right)
        @test inside[]
        @test SciMLBase.successful_retcode(sol)
        @test all(isfinite, (sol.u, sol.resid, left, right))
        @test lo <= left <= reference <= right <= hi
        @test iszero(sol.resid) || nextfloat(left) == right
    end
end

@testset "Large decimal bracket with zero endpoint derivatives" begin
    a, b = 1.0e308, 1.7e308
    inside = Ref(true)
    f = function (x, p)
        inside[] &= isfinite(x) && a <= x <= b
        s = (x - a) / (b - a)
        return 3s^2 - 2s^3 - 0.3
    end
    df(x, p) = (s = (x - a) / (b - a); 6s * (1 - s) / (b - a))
    sol = solve(
        IntervalNonlinearProblem(IntervalNonlinearFunction(f; jac = df), (a, b)),
        NewtonBisection(); abstol = 0.0
    )
    @test inside[]
    @test SciMLBase.successful_retcode(sol)
    @test all(isfinite, (sol.u, sol.resid, sol.left, sol.right))
    @test a <= sol.left <= sol.u <= sol.right <= b
    @test f(sol.left, nothing) <= 0 <= f(sol.right, nothing)
end

@testset "Undefined residual preserves the enclosing bracket" begin
    @testset "$alg, reverse=$rev, point=$bad" for alg in (NewtonBisection(),),
            rev in (false, true), bad in (0.0, 1.0, 0.5, 0.75)
        root = 0.875
        f(x, p) = x == bad ? NaN : x - root
        prob = IntervalNonlinearProblem(
            IntervalNonlinearFunction(f; jac = (x, p) -> 0.0), rev ? (1.0, 0.0) : (0.0, 1.0)
        )
        sol = solve(prob, alg; abstol = 0.0)
        expected = bad == 0.75 ? (0.5, 1.0) : (0.0, 1.0)
        @test sol.retcode == ReturnCode.Failure
        @test !SciMLBase.successful_retcode(sol)
        @test sol.u == bad
        @test isnan(sol.resid)
        @test (sol.left, sol.right) == (rev ? reverse(expected) : expected)
        @test min(sol.left, sol.right) <= root <= max(sol.left, sol.right)
    end
end
