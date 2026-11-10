using NonlinearSolveFirstOrder, SciMLBase, SparseArrays
using LineSearch: BackTracking

# Underdetermined NLLS with an analytic `jac` that writes only the stored values of the
# sparse `jac_prototype` (as ModelingToolkit generates it). The line search uses the
# JVP/VJP operators, so their Jacobian cache must also be sparse.
function f_sparse!(r, u, p)
    r[1] = u[1]^2 + u[2] - p[1]
    r[2] = u[2] * u[3] - p[2]
    return nothing
end
function j_sparse!(J, u, p)
    nz = J.nzval
    # CSC order of the pattern [1 1 0; 0 1 1]: (1,1), (1,2), (2,2), (2,3)
    nz[1] = 2u[1]
    nz[2] = 1.0
    nz[3] = u[3]
    nz[4] = u[2]
    return nothing
end
jp = sparse([1, 1, 2, 2], [1, 2, 2, 3], [1.0, 1.0, 1.0, 1.0], 2, 3)
nf = NonlinearFunction{true}(
    f_sparse!; jac = j_sparse!, jac_prototype = jp, resid_prototype = zeros(2)
)
prob = NonlinearLeastSquaresProblem(nf, [1.0, 1.0, 1.0], [2.0, 3.0])

@testset "$name" for (name, alg) in (
        ("GaussNewton", GaussNewton()),
        ("GaussNewton + BackTracking", GaussNewton(; linesearch = BackTracking())),
        ("LevenbergMarquardt", LevenbergMarquardt()),
        ("TrustRegion", TrustRegion()),
    )
    sol = solve(prob, alg; abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    @test maximum(abs, sol.resid) < 1.0e-8
end
