using JLArrays, NonlinearSolveFirstOrder

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
finally
    JLArrays.allowscalar(true)
end
