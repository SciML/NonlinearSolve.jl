using BracketingNonlinearSolve, SciMLBase, Test, ForwardDiff
using BracketingNonlinearSolve: NonlinearVerbosity
using SciMLLogging: None, Minimal, Standard

# Every bracketing method accepts `verbose` as `NonlinearSolveBase.solve` does (a `NonlinearVerbosity`,
# a preset or a `Bool`), also from the problem's kwargs.
f(u, p) = u^2 - p
prob = IntervalNonlinearProblem{false}(f, (0.0, 2.0), 2.0)
const ALGS = (ModAB(), Bisection(), ITP(), Ridder(), Falsi(), Brent(), Alefeld())
for alg in ALGS
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

# a solve keyword overrides the problem's `verbose`, in both directions
@test_logs (:warn,) match_mode = :any solve(remake(bad; kwargs = (; verbose = None())), ModAB(); verbose = Standard())
@test_logs solve(remake(bad; kwargs = (; verbose = Standard())), ModAB(); verbose = None())

# ForwardDiff through the parameter, with a preset: the root of u^2 - p is sqrt(p), d/dp = 1 / (2 sqrt(p))
for alg in ALGS
    g(p) = solve(IntervalNonlinearProblem{false}(f, (0.0, 2.0), p), alg; verbose = None()).u
    @test g(2.0) ≈ sqrt(2.0) atol = 1.0e-8
    @test ForwardDiff.derivative(g, 2.0) ≈ inv(2sqrt(2.0)) atol = 1.0e-6
    h(p) = solve(remake(IntervalNonlinearProblem{false}(f, (0.0, 2.0), p); kwargs = (; verbose = false)), alg).u
    @test ForwardDiff.derivative(h, 2.0) ≈ inv(2sqrt(2.0)) atol = 1.0e-6
end
