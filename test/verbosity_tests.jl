@testitem "Nonlinear Verbosity" tags = [:verbosity] begin
    using NonlinearSolve
    using BracketingNonlinearSolve
    using NonlinearSolve: NonlinearVerbosity
    using LinearSolve: LinearVerbosity
    using SciMLLogging: SciMLLogging
    using Test

    @testset "NonlinearVerbosity preset constructors" begin
        v_none = NonlinearVerbosity(SciMLLogging.None())
        v_all = NonlinearVerbosity(SciMLLogging.All())
        v_minimal = NonlinearVerbosity(SciMLLogging.Minimal())
        v_standard = NonlinearVerbosity(SciMLLogging.Standard())
        v_detailed = NonlinearVerbosity(SciMLLogging.Detailed())

        @test v_none.non_enclosing_interval isa SciMLLogging.Silent
        @test v_none.threshold_state isa SciMLLogging.Silent
        @test v_none.alias_u0_immutable isa SciMLLogging.Silent
        @test v_none.sensitivity_vjp_choice isa SciMLLogging.Silent

        @test v_minimal.non_enclosing_interval isa SciMLLogging.WarnLevel
        @test v_minimal.alias_u0_immutable isa SciMLLogging.Silent
        @test v_minimal.termination_condition isa SciMLLogging.Silent
        @test v_minimal.sensitivity_vjp_choice isa SciMLLogging.Silent

        @test v_standard.non_enclosing_interval isa SciMLLogging.WarnLevel
        @test v_standard.threshold_state isa SciMLLogging.WarnLevel
        @test v_standard.sensitivity_vjp_choice isa SciMLLogging.WarnLevel

        @test v_detailed.alias_u0_immutable isa SciMLLogging.WarnLevel
        @test v_detailed.termination_condition isa SciMLLogging.WarnLevel
        @test v_detailed.sensitivity_vjp_choice isa SciMLLogging.WarnLevel

        @test v_all.linsolve_failed_noncurrent isa SciMLLogging.WarnLevel
        @test v_all.threshold_state isa SciMLLogging.InfoLevel
        @test v_all.sensitivity_vjp_choice isa SciMLLogging.WarnLevel
    end

    @testset "Group-level keyword constructors" begin
        v_error = NonlinearVerbosity(error_control = SciMLLogging.ErrorLevel())
        @test v_error.alias_u0_immutable isa SciMLLogging.ErrorLevel
        @test v_error.non_enclosing_interval isa SciMLLogging.ErrorLevel
        @test v_error.termination_condition isa SciMLLogging.ErrorLevel
        @test v_error.linsolve_failed_noncurrent isa SciMLLogging.ErrorLevel

        v_numerical = NonlinearVerbosity(numerical = SciMLLogging.Silent())
        @test v_numerical.threshold_state isa SciMLLogging.Silent

        v_sensitivity = NonlinearVerbosity(sensitivity = SciMLLogging.Silent())
        @test v_sensitivity.sensitivity_vjp_choice isa SciMLLogging.Silent

        v_sensitivity2 = NonlinearVerbosity(sensitivity = SciMLLogging.InfoLevel())
        @test v_sensitivity2.sensitivity_vjp_choice isa SciMLLogging.InfoLevel
    end

    @testset "Mixed group and individual settings" begin
        v_mixed = NonlinearVerbosity(
            numerical = SciMLLogging.Silent(),
            threshold_state = SciMLLogging.WarnLevel(),
            error_control = SciMLLogging.InfoLevel()
        )
        # Individual override should take precedence
        @test v_mixed.threshold_state isa SciMLLogging.WarnLevel
        # Error control group setting should apply
        @test v_mixed.alias_u0_immutable isa SciMLLogging.InfoLevel
        @test v_mixed.linsolve_failed_noncurrent isa SciMLLogging.InfoLevel
    end

    @testset "Individual keyword arguments" begin
        v_individual = NonlinearVerbosity(
            alias_u0_immutable = SciMLLogging.ErrorLevel(),
            threshold_state = SciMLLogging.InfoLevel(),
            termination_condition = SciMLLogging.Silent()
        )
        @test v_individual.alias_u0_immutable isa SciMLLogging.ErrorLevel
        @test v_individual.threshold_state isa SciMLLogging.InfoLevel
        @test v_individual.termination_condition isa SciMLLogging.Silent
        # Unspecified options should use defaults
        @test v_individual.non_enclosing_interval isa SciMLLogging.WarnLevel
        @test v_individual.linsolve_failed_noncurrent isa SciMLLogging.WarnLevel
    end

    g(u, p) = u^2 - 4

    int_prob = IntervalNonlinearProblem(g, (3.0, 5.0))

    @test_logs (
        :info,
        r"The interval is not an enclosing interval, opposite signs at the boundaries are required.",
    ) solve(
        int_prob,
        ITP(), verbose = NonlinearVerbosity(non_enclosing_interval = SciMLLogging.InfoLevel())
    )

    @test_logs (
        :error,
        r"The interval is not an enclosing interval, opposite signs at the boundaries are required.",
    ) @test_throws ErrorException solve(
        int_prob,
        ITP(), verbose = NonlinearVerbosity(non_enclosing_interval = SciMLLogging.ErrorLevel())
    )

    # Test that the linear verbosity is passed to the linear solve
    f(u, p) = [u[1]^2 - 2u[1] + 1, sum(u)]
    prob = NonlinearProblem(f, [1.0, 1.0])

    @test_logs (
        :warn,
        r"LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.",
    ) match_mode = :any solve(
        prob,
        verbose = NonlinearVerbosity(linear_verbosity = SciMLLogging.Detailed())
    )

    @test_logs (
        :info,
        r"LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.",
    ) match_mode = :any solve(
        prob,
        verbose = NonlinearVerbosity(linear_verbosity = LinearVerbosity(default_lu_fallback = SciMLLogging.InfoLevel()))
    )

    @test_logs min_level = 0 solve(
        prob,
        verbose = NonlinearVerbosity(linear_verbosity = SciMLLogging.Standard())
    )

    @test_logs (
        :warn,
        r"LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.",
    ) match_mode = :any solve(
        prob,
        verbose = NonlinearVerbosity(linear_verbosity = SciMLLogging.Detailed())
    )

    @test_logs min_level = 0 solve(
        prob,
        verbose = NonlinearVerbosity(SciMLLogging.None())
    )

    @test_logs min_level = 0 solve(
        prob,
        verbose = false
    )

    # Test that caches get correct verbosities
    cache = init(
        prob, verbose = NonlinearVerbosity(threshold_state = SciMLLogging.InfoLevel())
    )

    @test cache.verbose.threshold_state == SciMLLogging.InfoLevel()

    f(u, p) = u .* u .- 2
    prob = NonlinearProblem(f, [1.0, 1.0])

    @testset "solve with Bool verbose" begin
        # Test verbose = true works (default verbosity)
        sol1 = solve(prob, verbose = true)
        @test sol1.retcode == ReturnCode.Success

        # Test verbose = false silences all output
        @test_logs min_level = 0 sol2 = solve(prob, verbose = false)
    end

    @testset "solve with Preset verbose" begin
        # Test verbose = SciMLLogging.Standard() works
        sol1 = solve(prob, verbose = SciMLLogging.Standard())
        @test sol1.retcode == ReturnCode.Success

        # Test verbose = SciMLLogging.None() silences output
        @test_logs min_level = 0 sol2 = solve(prob, verbose = SciMLLogging.None())

        # Test verbose = SciMLLogging.Detailed() works
        sol3 = solve(prob, verbose = SciMLLogging.Detailed())
        @test sol3.retcode == ReturnCode.Success

        # Test verbose = SciMLLogging.All() works
        sol4 = solve(prob, verbose = SciMLLogging.All())
        @test sol4.retcode == ReturnCode.Success

        # Test verbose = SciMLLogging.Minimal() works
        sol5 = solve(prob, verbose = SciMLLogging.Minimal())
        @test sol5.retcode == ReturnCode.Success
    end

    @testset "init with Bool verbose" begin
        # Test verbose = true converts to NonlinearVerbosity()
        cache1 = init(prob, verbose = true)
        @test cache1.verbose isa NonlinearVerbosity
        @test cache1.verbose.threshold_state == SciMLLogging.WarnLevel()

        # Test verbose = false converts to NonlinearVerbosity(None())
        cache2 = init(prob, verbose = false)
        @test cache2.verbose isa NonlinearVerbosity
        @test cache2.verbose.threshold_state isa SciMLLogging.Silent
        @test cache2.verbose.non_enclosing_interval isa SciMLLogging.Silent
    end

    @testset "init with Preset verbose" begin
        # Test verbose = SciMLLogging.Standard() converts to NonlinearVerbosity(Standard())
        cache1 = init(prob, verbose = SciMLLogging.Standard())
        @test cache1.verbose isa NonlinearVerbosity
        @test cache1.verbose.threshold_state == SciMLLogging.WarnLevel()

        # Test verbose = SciMLLogging.None() converts to NonlinearVerbosity(None())
        cache2 = init(prob, verbose = SciMLLogging.None())
        @test cache2.verbose isa NonlinearVerbosity
        @test cache2.verbose.threshold_state isa SciMLLogging.Silent

        # Test verbose = SciMLLogging.Detailed()
        cache3 = init(prob, verbose = SciMLLogging.Detailed())
        @test cache3.verbose isa NonlinearVerbosity
        @test cache3.verbose.linear_verbosity isa SciMLLogging.Detailed

        # Test verbose = SciMLLogging.All()
        cache4 = init(prob, verbose = SciMLLogging.All())
        @test cache4.verbose isa NonlinearVerbosity
        @test cache4.verbose.threshold_state == SciMLLogging.InfoLevel()

        # Test verbose = SciMLLogging.Minimal()
        cache5 = init(prob, verbose = SciMLLogging.Minimal())
        @test cache5.verbose isa NonlinearVerbosity
        @test cache5.verbose.alias_u0_immutable isa SciMLLogging.Silent
    end

    @testset "init then solve with converted verbose" begin
        # Ensure the converted verbose works through the full solve pipeline
        cache = init(prob, verbose = false)
        @test_logs min_level = 0 sol = solve!(cache)
    end
end
