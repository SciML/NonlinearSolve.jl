using BracketingNonlinearSolve, Unitful
import DynamicQuantities as DQ

const get_tolerance = BracketingNonlinearSolve.NonlinearSolveBase.get_tolerance

@testset "Default and zero tolerances take the units of the bracket" begin
    for x in (0.5u"s", 0.5 * DQ.u"s")
        T = typeof(x)
        @test get_tolerance(x, nothing, T) ≈ 3.0e-13 * oneunit(x) rtol = 1.0e-3
        @test get_tolerance(x, 0.0, T) == zero(x)
        @test get_tolerance(x, 1.0e-9 * oneunit(x), T) == 1.0e-9 * oneunit(x)
    end
end

@testset "$(nameof(typeof(alg))) with units" for alg in (
        Alefeld(), Bisection(), Brent(), Falsi(), ITP(), Muller(), Ridder(), ModAB(),
    )
    # Units in the residual only
    sol = solve(IntervalNonlinearProblem((x, p) -> (x - 0.3) * u"m", (0.0, 1.0)), alg)
    @test sol.u ≈ 0.3

    # Units in the bracket and the residual
    for (s, m) in ((u"s", u"m"), (DQ.u"s", DQ.u"m"))
        f = (x, p) -> (x^2 - 2s^2) * m
        # With a zero tolerance Muller stops only at an exact root, which √2 has not
        for abstol in (alg isa Muller ? (nothing,) : (nothing, 0.0))
            sol = solve(IntervalNonlinearProblem(f, (0.0s, 2.0s)), alg; abstol)
            @test sol.u ≈ sqrt(2.0) * s
        end
    end
end
