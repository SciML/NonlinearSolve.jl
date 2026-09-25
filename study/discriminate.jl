# The "Evaluations" tests of test/rootfind_tests__item9.jl, run against the NewtonBisection
# of PR 1330 (PRNewtonBisection, study/pr1330_newton_bisection.jl) and against this branch.
#   julia --project=study study/discriminate.jl
using BracketingNonlinearSolve, Test
include("pr1330_newton_bisection.jl")

function count_calls(alg, f, df, tspan)
    fcalls, jcalls = Ref(0), Ref(0)
    fun = IntervalNonlinearFunction(
        (u, p) -> (fcalls[] += 1; f(u)); jac = (u, p) -> (jcalls[] += 1; df(u))
    )
    solve(IntervalNonlinearProblem(fun, tspan), alg; abstol = 0.0)
    return fcalls[], jcalls[]
end

@testset "PR 1330 versus this branch" begin
    @testset "$(nameof(typeof(alg)))" for alg in (PRNewtonBisection(), NewtonBisection())
        begin
            @testset "multiple roots: fewer evaluations than ModAB" for m in (3, 5)
                f, df = u -> (u - 1)^m, u -> m * (u - 1)^(m - 1)
                nf, _ = count_calls(alg, f, df, (0.0, 3.0))
                nmodab, _ = count_calls(ModAB(), f, df, (0.0, 3.0))
                @test nf < nmodab
            end
            @testset "fewer derivatives than f" for (f, df, tspan) in (
                    (u -> u^2 - 2, u -> 2u, (1.0, 20.0)),
                    (u -> exp(u) - 10, u -> exp(u), (0.0, 5.0)),
                    (u -> cos(u) - u, u -> -sin(u) - 1, (0.0, 1.0)),
                )
                nf, nj = count_calls(alg, f, df, tspan)
                @test nj < nf
            end
        end
    end
end
