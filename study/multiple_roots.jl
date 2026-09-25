# Multiple roots away from exactly representable points, abstol = 0: "f/jac" counts.
#   julia --project=study study/multiple_roots.jl
using BracketingNonlinearSolve, ForwardDiff
include("pr1330_newton_bisection.jl")
const C = zeros(Int, 2)
for (name, f, df, br) in (
        ("(u - 1/3)^3", u -> (u - 1 / 3)^3, u -> 3(u - 1 / 3)^2, (0.0, 3.0)),
        ("(u - π/4)^3", u -> (u - π / 4)^3, u -> 3(u - π / 4)^2, (0.0, 3.0)),
        ("(u - π/4)^5", u -> (u - π / 4)^5, u -> 5(u - π / 4)^4, (0.0, 3.0)),
        ("(u - π/4)^3 expanded", u -> ((u - 3 * (π / 4)) * u + 3 * (π / 4)^2) * u - (π / 4)^3, u -> 3u^2 - 6(π / 4) * u + 3(π / 4)^2, (0.0, 3.0)),
        ("sin(u - π/4)^3", u -> sin(u - π / 4)^3, u -> 3sin(u - π / 4)^2 * cos(u - π / 4), (0.0, 3.0)),
        ("(exp(u) - 2)^3", u -> (exp(u) - 2)^3, u -> 3(exp(u) - 2)^2 * exp(u), (0.0, 3.0)),
    )
    print(rpad(name, 24))
    for (an, alg, jac) in (("ModAB", ModAB(), false), ("PR NB", PRNewtonBisection(), true), ("NB", NewtonBisection(), true), ("ModABNewton", ModABNewton(), true))
        fill!(C, 0)
        fc = (u, p) -> (C[1] += 1; f(u))
        fun = jac ? IntervalNonlinearFunction(fc; jac = (u, p) -> (C[2] += 1; df(u))) : IntervalNonlinearFunction(fc)
        sol = solve(IntervalNonlinearProblem(fun, br), alg; abstol = 0.0)
        print(" | ", an, " ", C[1], "/", C[2], " ", sol.retcode, " w=", sol.right - sol.left)
    end
    println()
end
