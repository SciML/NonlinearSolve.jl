@testitem "Nonlinear Verbosity" tags=[:misc] begin
    using IntervalNonlinearSolve
    using NonlinearSolve
    using NonlinearSolve: NonlinearVerbosity
    using LinearSolve: LinearVerbosity
    using SciMLVerbosity: SciMLVerbosity, Verbosity
    using Test 
    using Logging

    g(u, p) = u^2 - 4

    int_prob = IntervalNonlinearProblem(g, (3.0, 5.0))

    @test_logs (:info,
        "The interval is not an enclosing interval, opposite signs at the boundaries are required.") solve(
        int_prob,
        ITP(), verbose = NonlinearVerbosity(non_enclosing_interval = Verbosity.Info()))

    @test_logs (:error,
        "The interval is not an enclosing interval, opposite signs at the boundaries are required.") solve(
        int_prob,
        ITP(), verbose = NonlinearVerbosity(non_enclosing_interval = Verbosity.Error()))

    # Test that the linear verbosity is passed to the linear solve
    f(u, p) = [u[1]^2 - 2u[1] + 1, sum(u)]
    prob = NonlinearProblem(f, [1.0, 1.0])

    @test_logs (:warn,
        "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") match_mode=:any solve(
        prob,
        verbose = NonlinearVerbosity())

    @test_logs (:info,
        "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") match_mode=:any solve(
        prob,
        verbose = NonlinearVerbosity(linear_verbosity = LinearVerbosity(default_lu_fallback = Verbosity.Info())))

    @test_logs (:warn,
        "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") match_mode=:any solve(
        prob,
        verbose = true)

    @test_logs min_level=Logging.Info solve(prob,
        verbose = NonlinearVerbosity(Verbosity.None()))

    @test_logs min_level=Logging.Info solve(prob,
        verbose = false)

    # Test that caches get correct verbosities
    cache = init(prob, verbose = NonlinearVerbosity(threshold_state = Verbosity.Info()))

    @test cache.verbose.numerical.threshold_state == Verbosity.Info()
end
