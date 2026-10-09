using SciMLJacobianOperators

using SciMLBase, LinearAlgebra, SparseArrays

# The analytic `jac` writes only the stored values of the sparse `jac_prototype`, as the
# ModelingToolkit-generated sparse Jacobians do. A dense cache has no `nzval` field.
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
j_dense(u) = [2u[1] 1.0 0.0; 0.0 u[3] u[2]]

@testset "jac_prototype = $(typeof(jp))" for jp in (
        sparse([1, 1, 2, 2], [1, 2, 2, 3], [1.0, 1.0, 1.0, 1.0], 2, 3),
        sparse([1, 1, 2, 2], [1, 2, 2, 3], trues(4), 2, 3),
    )
    nf = NonlinearFunction{true}(
        f_sparse!; jac = j_sparse!, jac_prototype = jp, resid_prototype = zeros(2)
    )
    u0, p = [1.0, 2.0, 3.0], [2.0, 3.0]
    prob = NonlinearLeastSquaresProblem(nf, u0, p)
    fu0 = zeros(2)
    f_sparse!(fu0, u0, p)

    jac_op = JacobianOperator(prob, fu0, u0)
    sop = StatefulJacobianOperator(jac_op, u0, p)
    J = j_dense(u0)

    v = [0.3, -0.7, 1.1]
    w = [0.4, -0.9]
    @test sop * v ≈ J * v
    @test sop' * w ≈ J' * w
    Jv = zeros(2)
    vJ = zeros(3)
    mul!(Jv, sop, v)
    mul!(vJ, sop', w)
    @test Jv ≈ J * v
    @test vJ ≈ J' * w
    @test (sop' * sop) * v ≈ J' * (J * v)
end

# A `jac` that writes only the nonzero entries needs a zero cache: the cache is reused, so
# entries that `jac` never writes keep their first value.
@testset "analytic jac cache starts at zero" begin
    u0, fu0 = [1.0, 2.0, 3.0], zeros(2)
    for jp in (
            nothing, zeros(2, 3),
            sparse([1, 1, 2, 2], [1, 2, 2, 3], [1.0, 1.0, 1.0, 1.0], 2, 3),
            sparse([1, 1, 2, 2], [1, 2, 2, 3], trues(4), 2, 3),
        )
        nf = NonlinearFunction{true}(
            f_sparse!; jac = j_sparse!, jac_prototype = jp, resid_prototype = fu0
        )
        cache = SciMLJacobianOperators.analytic_jac_cache(nf, u0, fu0)
        @test eltype(cache) == Float64
        @test size(cache) == (2, 3)
        @test iszero(cache)
        jp isa SparseMatrixCSC && @test cache isa SparseMatrixCSC && nnz(cache) == nnz(jp)
    end

    function j_nonzeros!(J, u, p)
        J[1, 1] = 2u[1]
        J[1, 2] = 1.0
        J[2, 2] = u[3]
        J[2, 3] = u[2]
        return nothing
    end
    nf = NonlinearFunction{true}(f_sparse!; jac = j_nonzeros!, resid_prototype = fu0)
    p = [2.0, 3.0]
    prob = NonlinearLeastSquaresProblem(nf, u0, p)
    sop = StatefulJacobianOperator(JacobianOperator(prob, fu0, u0), u0, p)
    J = j_dense(u0)
    v = [0.3, -0.7, 1.1]
    w = [0.4, -0.9]
    @test sop * v ≈ J * v
    @test sop' * w ≈ J' * w
end
