using NonlinearSolveBase
using NonlinearSolveBase: dampen_jacobian!!
using ArrayInterface
using LinearAlgebra
using StaticArrays
using Test

# `dampen_jacobian!!` has a mutable path (`can_setindex(J_cache)`, damping written in place) and
# an out-of-place fallback for Jacobians that cannot `setindex!` — `SMatrix`, and GPU sparse
# matrices, which are mutable only through `nonzeros`. The two must agree: damping is defined by
# the math, not by how the Jacobian happens to be stored. The scalar-`D` fallback used to read
# `J .+ D`, which adds `D` to *every* entry rather than only the diagonal, so an `SMatrix`
# Jacobian was silently damped wrong (a convergence bug, not a wrong root: damping does not move
# the fixed point).

J = [1.0 2.0; 3.0 4.0]
Jstatic = SMatrix{2, 2}(J)

@testset "scalar damping touches only the diagonal" begin
    D = 10.0
    expected = J + D * I   # [11 2; 3 14]

    @test dampen_jacobian!!(copy(J), J, D) == expected
    @test !ArrayInterface.can_setindex(Jstatic)   # ensure we exercise the fallback
    @test Array(dampen_jacobian!!(Jstatic, Jstatic, D)) == expected

    # The off-diagonal entries must survive untouched.
    damped = dampen_jacobian!!(Jstatic, Jstatic, D)
    @test damped[1, 2] == J[1, 2]
    @test damped[2, 1] == J[2, 1]
end

@testset "diagonal damping touches only the diagonal" begin
    D = Diagonal([10.0, 20.0])
    expected = J + D

    @test dampen_jacobian!!(copy(J), J, D) == expected
    @test Array(dampen_jacobian!!(Jstatic, Jstatic, D)) == expected
end

@testset "matrix damping adds the full matrix" begin
    D = [10.0 100.0; 200.0 20.0]
    expected = J + D

    @test dampen_jacobian!!(copy(J), J, D) == expected
    @test Array(dampen_jacobian!!(Jstatic, Jstatic, D)) == expected
end

@testset "mutable and immutable paths agree" begin
    # The invariant behind all of the above, stated directly.
    for D in (10.0, Diagonal([10.0, 20.0]), [10.0 100.0; 200.0 20.0])
        @test Array(dampen_jacobian!!(Jstatic, Jstatic, D)) ==
            dampen_jacobian!!(copy(J), J, D)
    end
end
