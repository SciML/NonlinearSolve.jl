using BracketingNonlinearSolve

# Counts the evaluations of f in a solve with abstol = 0.
function count_f(alg, f, df, tspan)
    calls = Ref(0)
    fun = IntervalNonlinearFunction((u, p) -> (calls[] += 1; f(u)); jac = (u, p) -> df(u))
    sol = solve(IntervalNonlinearProblem(fun, tspan), alg; abstol = 0.0)
    return calls[], sol
end

@testset "NewtonBisection at a root of multiplicity $m, $name" for m in (3, 5),
        (name, r) in (("1", 1.0), ("1/3", 1 / 3), ("π/4", π / 4))

    f = u -> (u - r)^m
    df = u -> m * (u - r)^(m - 1)
    n_newton, sol = count_f(NewtonBisection(), f, df, (0.0, 3.0))
    n_modab, _ = count_f(ModAB(), f, df, (0.0, 3.0))
    @test SciMLBase.successful_retcode(sol)
    @test iszero(f(sol.u)) || (sign(f(sol.left)) != sign(f(sol.right)) && nextfloat(sol.left) == sol.right)
    # without the step correction Newton converges linearly here, needing about 90 evaluations
    @test n_newton < n_modab
end

@testset "NewtonBisection at a multiple root of a non-polynomial" for (f, df) in (
        (u -> sin(u - π / 4)^3, u -> 3sin(u - π / 4)^2 * cos(u - π / 4)),
        (u -> (exp(u) - 2)^3, u -> 3(exp(u) - 2)^2 * exp(u)),
    )
    n, sol = count_f(NewtonBisection(), f, df, (0.0, 3.0))
    @test SciMLBase.successful_retcode(sol)
    @test iszero(f(sol.u)) || (sign(f(sol.left)) != sign(f(sol.right)) && nextfloat(sol.left) == sol.right)
    @test n ≤ 20
end
