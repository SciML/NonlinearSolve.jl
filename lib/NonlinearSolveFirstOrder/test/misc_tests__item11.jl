using JLArrays, NonlinearSolveFirstOrder, NonlinearSolveBase, LinearSolve, Logging
using ADTypes: AutoForwardDiff, AutoPolyesterForwardDiff
using SciMLBase: NLStats

JLArrays.allowscalar(false)
try
    jlarray_f(u, p) = u .^ 2 .- p
    jlarray_prob = NonlinearProblem(jlarray_f, JLArray([1.0, 2.0]), 4.0)

    jlarray_nr = solve(jlarray_prob, NewtonRaphson())
    @test jlarray_nr.retcode == ReturnCode.Success
    @test jlarray_nr.u isa JLArray
    @test Array(jlarray_nr.u) ≈ [2.0, 2.0]

    jlarray_tr = solve(jlarray_prob, TrustRegion())
    @test jlarray_tr.retcode == ReturnCode.Success
    @test jlarray_tr.u isa JLArray
    @test Array(jlarray_tr.u) ≈ [2.0, 2.0]

    # Matrix-free Krylov linsolve never materializes a concrete Jacobian, routing
    # through `JacobianOperator`'s eagerly-prepared VJP/JVP instead of `DI.jacobian`.
    # ForwardDiff pushforwards do not scalar-index, so they stay the JVP backend.
    jlarray_nr_krylov = solve(jlarray_prob, NewtonRaphson(linsolve = KrylovJL_GMRES()))
    @test jlarray_nr_krylov.retcode == ReturnCode.Success
    @test Array(jlarray_nr_krylov.u) ≈ [2.0, 2.0]
    @test jlarray_nr_krylov.alg.jvp_autodiff isa AutoForwardDiff

    @testset "Explicit ForwardDiff on the matrix-free path: $(kw)" for kw in (
            (; jvp_autodiff = AutoForwardDiff()), (; autodiff = AutoForwardDiff()),
        )
        alg = NewtonRaphson(; linsolve = KrylovJL_GMRES(), kw...)
        sol = @test_logs min_level = Logging.Warn solve(jlarray_prob, alg)
        @test sol.retcode == ReturnCode.Success
        @test Array(sol.u) ≈ [2.0, 2.0]
        @test sol.alg.jvp_autodiff isa AutoForwardDiff
    end

    jlarray_tr_krylov = solve(jlarray_prob, TrustRegion(linsolve = KrylovJL_GMRES()))
    @test jlarray_tr_krylov.retcode == ReturnCode.Success
    @test Array(jlarray_tr_krylov.u) ≈ [2.0, 2.0]

    # Yuan/Bastin radius updates eagerly compute `Jᵀfu` through the VJP operator at
    # `init` time, so they exercise the DI pullback path outside the Krylov cache too.
    jlarray_tr_yuan = solve(
        jlarray_prob, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Yuan)
    )
    @test jlarray_tr_yuan.retcode == ReturnCode.Success
    @test Array(jlarray_tr_yuan.u) ≈ [2.0, 2.0]

    jlarray_tr_bastin = solve(
        jlarray_prob, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Bastin)
    )
    @test jlarray_tr_bastin.retcode == ReturnCode.Success
    @test Array(jlarray_tr_bastin.u) ≈ [2.0, 2.0]

    jlarray_fu = jlarray_f(jlarray_prob.u0, jlarray_prob.p)
    @testset "Explicit $(nameof(typeof(ad))) Jacobian falls back to finite differencing" for ad in (
            AutoForwardDiff(), AutoPolyesterForwardDiff(),
        )
        jac_cache = @test_logs (:warn, r"scalar indexing") NonlinearSolveBase.construct_jacobian_cache(
            jlarray_prob, NewtonRaphson(), jlarray_prob.f, jlarray_fu;
            stats = NLStats(0, 0, 0, 0, 0), autodiff = ad
        )
        J = NonlinearSolveBase.reused_jacobian(jac_cache, jlarray_prob.u0)
        @test J isa JLArray
        # Forward differences carry an O(sqrt(eps)) truncation error.
        @test Array(J) ≈ [2.0 0.0; 0.0 4.0] rtol = 1.0e-6
    end

    jlarray_nr_forwarddiff = @test_logs (:warn, r"scalar indexing") solve(
        jlarray_prob, NewtonRaphson(autodiff = AutoForwardDiff())
    )
    @test jlarray_nr_forwarddiff.retcode == ReturnCode.Success
    @test Array(jlarray_nr_forwarddiff.u) ≈ [2.0, 2.0]
finally
    JLArrays.allowscalar(true)
end
