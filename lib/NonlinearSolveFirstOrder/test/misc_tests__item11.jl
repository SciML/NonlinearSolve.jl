using JLArrays, NonlinearSolveFirstOrder, LinearSolve

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
    jlarray_nr_krylov = solve(jlarray_prob, NewtonRaphson(linsolve = KrylovJL_GMRES()))
    @test jlarray_nr_krylov.retcode == ReturnCode.Success
    @test Array(jlarray_nr_krylov.u) ≈ [2.0, 2.0]

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
finally
    JLArrays.allowscalar(true)
end
