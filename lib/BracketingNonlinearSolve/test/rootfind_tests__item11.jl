using BracketingNonlinearSolve, SciMLBase, Test
using BracketingNonlinearSolve: NonlinearVerbosity
using SciMLLogging: None, Minimal, Standard

# Every bracketing method accepts `verbose` as `NonlinearSolveBase.solve` does (a `NonlinearVerbosity`,
# a preset or a `Bool`), also from the problem's kwargs.
f(u, p) = u^2 - p
prob = IntervalNonlinearProblem{false}(f, (0.0, 2.0), 2.0)
for alg in (ModAB(), Bisection(), ITP(), Ridder(), Falsi(), Brent(), Alefeld())
    ref = solve(prob, alg)
    for v in (NonlinearVerbosity(), NonlinearVerbosity(None()), None(), Minimal(), Standard(), true, false)
        sol = solve(prob, alg; verbose = v)
        @test sol.retcode == ref.retcode
        @test sol.u === ref.u
    end
    sol = solve(remake(prob; kwargs = (; verbose = None())), alg)
    @test sol.u === ref.u
end
# a non-enclosing bracket warns at the default verbosity and is a quiet failure with logging off
bad = IntervalNonlinearProblem{false}(f, (2.0, 3.0), 2.0)
@test_logs (:warn,) match_mode = :any solve(bad, ModAB(); verbose = Standard())
@test_logs solve(bad, ModAB(); verbose = None())
@test solve(bad, ModAB(); verbose = None()).retcode == ReturnCode.InitialFailure
