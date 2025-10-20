@testitem "Nonlinear Verbosity" tags=[:misc] begin
    using IntervalNonlinearSolve
    using NonlinearSolve
    using NonlinearSolve: NonlinearVerbosity
    using LinearSolve: LinearVerbosity
    using SciMLLogging: SciMLLogging
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
        verbose = NonlinearVerbosity(linear_verbosity = Detailed()))

    @test_logs (:info,
        "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") match_mode=:any solve(
        prob,
        verbose = NonlinearVerbosity(linear_verbosity = LinearVerbosity(default_lu_fallback = Verbosity.Info())))

    @test_logs min_level=Logging.Info solve(prob,
        verbose = NonlinearVerbosity(Verbosity.None()))

    @test_logs min_level=Logging.Info solve(prob,
        verbose = false)

    # Test that caches get correct verbosities
    cache = init(prob, verbose = NonlinearVerbosity(threshold_state = Verbosity.Info()))

    @test cache.verbose.numerical.threshold_state == Verbosity.Info()

    @testset "NonlinearVerbosity preset constructors" begin
        v_none = NonlinearVerbosity(SciMLLogging.None())
        v_all = NonlinearVerbosity(SciMLLogging.All())
        v_minimal = NonlinearVerbosity(SciMLLogging.Minimal())
        v_standard = NonlinearVerbosity(SciMLLogging.Standard())
        v_detailed = NonlinearVerbosity(SciMLLogging.Detailed())

        @test v_none.immutable_u0 isa SciMLLogging.Silent
        @test v_none.threshold_state isa SciMLLogging.Silent
        @test v_none.colorvec_non_sparse isa SciMLLogging.Silent

        @test v_minimal.immutable_u0 isa SciMLLogging.ErrorLevel
        @test v_minimal.colorvec_non_sparse isa SciMLLogging.Silent
        @test v_minimal.non_forward_mode isa SciMLLogging.Silent

        @test v_standard.immutable_u0 isa SciMLLogging.WarnLevel
        @test v_standard.threshold_state isa SciMLLogging.WarnLevel

        @test v_detailed.colorvec_non_sparse isa SciMLLogging.InfoLevel
        @test v_detailed.non_forward_mode isa SciMLLogging.InfoLevel
        @test v_detailed.jacobian_free isa SciMLLogging.InfoLevel

        @test v_all.colorvec_non_sparse isa SciMLLogging.InfoLevel
        @test v_all.threshold_state isa SciMLLogging.InfoLevel
    end

    @testset "Group-level keyword constructors" begin
        v_error = NonlinearVerbosity(error_control = ErrorLevel())
        @test v_error.immutable_u0 isa SciMLLogging.ErrorLevel
        @test v_error.non_enclosing_interval isa SciMLLogging.ErrorLevel
        @test v_error.termination_condition isa SciMLLogging.ErrorLevel

        v_numerical = NonlinearVerbosity(numerical = Silent())
        @test v_numerical.threshold_state isa SciMLLogging.Silent
        @test v_numerical.pinv_undefined isa SciMLLogging.Silent

        v_performance = NonlinearVerbosity(performance = InfoLevel())
        @test v_performance.colorvec_non_sparse isa SciMLLogging.InfoLevel
        @test v_performance.colorvec_no_prototype isa SciMLLogging.InfoLevel
        @test v_performance.sparsity_using_jac_prototype isa SciMLLogging.InfoLevel
        @test v_performance.sparse_matrixcolorings_not_loaded isa SciMLLogging.InfoLevel
    end

    @testset "Mixed group and individual settings" begin
        v_mixed = NonlinearVerbosity(
            numerical = Silent(),
            threshold_state = WarnLevel(),
            performance = InfoLevel()
        )
        # Individual override should take precedence
        @test v_mixed.threshold_state isa SciMLLogging.WarnLevel
        # Other numerical options should use group setting
        @test v_mixed.pinv_undefined isa SciMLLogging.Silent
        # Performance group setting should apply
        @test v_mixed.colorvec_non_sparse isa SciMLLogging.InfoLevel
        @test v_mixed.colorvec_no_prototype isa SciMLLogging.InfoLevel
    end

    @testset "Individual keyword arguments" begin
        v_individual = NonlinearVerbosity(
            immutable_u0 = ErrorLevel(),
            threshold_state = InfoLevel(),
            colorvec_non_sparse = Silent()
        )
        @test v_individual.immutable_u0 isa SciMLLogging.ErrorLevel
        @test v_individual.threshold_state isa SciMLLogging.InfoLevel
        @test v_individual.colorvec_non_sparse isa SciMLLogging.Silent
        # Unspecified options should use defaults
        @test v_individual.non_enclosing_interval isa SciMLLogging.WarnLevel
        @test v_individual.pinv_undefined isa SciMLLogging.WarnLevel
    end

    @testset "Linear verbosity passthrough" begin
        v_with_linear = NonlinearVerbosity(
            linear_verbosity = SciMLLogging.Detailed()
        )
        @test v_with_linear.linear_verbosity isa SciMLLogging.Detailed

        v_with_linear2 = NonlinearVerbosity(
            linear_verbosity = SciMLLogging.None()
        )
        @test v_with_linear2.linear_verbosity isa SciMLLogging.None
    end
end
