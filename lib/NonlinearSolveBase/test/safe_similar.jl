using NonlinearSolveBase
using NonlinearSolveBase: Utils
using ArrayInterface
using SciMLOperators
using SparseArrays
using StaticArrays
using Test

# `safe_similar` must hand back a *zeroed* buffer, never uninitialized memory: `similar` on a
# BigFloat array yields undef references (reading them throws), and a `jac_prototype`-derived
# Jacobian is consumed at descent init (NLLS normal form forms `transpose(J) * J`) before the
# first Jacobian evaluation overwrites it.

@testset "dense buffers are zeroed" begin
    @test all(iszero, Utils.safe_similar(rand(4)))
    @test all(iszero, Utils.safe_similar(rand(3, 3)))
    @test all(iszero, Utils.safe_similar(rand(4), 2, 3))
    @test all(iszero, Utils.safe_similar(rand(4), Float32))

    # BigFloat is the original motivation: unfilled, these entries are undef *references* and
    # `iszero` on them throws rather than returning garbage.
    @test all(iszero, Utils.safe_similar(BigFloat[1, 2, 3]))
    @test all(iszero, Utils.safe_similar(rand(3), BigFloat))
end

@testset "immutable inputs whose `similar` is mutable" begin
    # `similar(::SArray)` is an `MArray`, so the buffer we hand out is fillable even though the
    # input is not. Same story for ranges and other non-setindexable inputs.
    y = Utils.safe_similar(SA[1.0, 2.0, 3.0])
    @test y isa MArray
    @test all(iszero, y)

    @test all(iszero, Utils.safe_similar(1:5))
end

@testset "sparse buffers are zeroed through their stored values" begin
    A = sprand(20, 20, 0.2)
    B = Utils.safe_similar(A)
    @test all(iszero, nonzeros(B))
    # The structural pattern must survive: it is what the prototype exists to carry, and sparse
    # AD coloring decompresses into exactly these entries. (`zero(A)` would drop all of them,
    # which is why the zeroing goes through `nonzeros` rather than out-of-place.)
    @test nnz(B) == nnz(A)
    @test rowvals(B) == rowvals(A)
    @test SparseArrays.getcolptr(B) == SparseArrays.getcolptr(A)

    v = sprand(20, 0.2)
    @test all(iszero, nonzeros(Utils.safe_similar(v)))
    @test nnz(Utils.safe_similar(v)) == nnz(v)
end

# GPU sparse matrices (`CuSparseMatrixCSC`, ...) implement no `setindex!` at all, so `fill!`
# does not work on them, but they are mutable through `nonzeros` — which is all the zeroing
# needs. Stand in for them with a sparse type that behaves the same way, so this is covered
# off-GPU.
struct NoSetindexSparseMatrix{Tv, Ti} <: AbstractSparseMatrix{Tv, Ti}
    m::Int
    n::Int
    vals::Vector{Tv}
end
Base.size(A::NoSetindexSparseMatrix) = (A.m, A.n)
SparseArrays.nonzeros(A::NoSetindexSparseMatrix) = A.vals
function Base.similar(A::NoSetindexSparseMatrix{Tv, Ti}) where {Tv, Ti}
    return NoSetindexSparseMatrix{Tv, Ti}(A.m, A.n, similar(A.vals))
end
ArrayInterface.can_setindex(::Type{<:NoSetindexSparseMatrix}) = false

@testset "sparse buffers that cannot `setindex!` are still zeroed" begin
    A = NoSetindexSparseMatrix{Float64, Int}(4, 4, [1.0, 2.0, 3.0])
    @test !ArrayInterface.can_setindex(A)
    @test all(iszero, nonzeros(Utils.safe_similar(A)))
end

# An array we cannot zero at all: no `setindex!`, and no `nonzeros` to reach into either.
struct UnfillableArray{T} <: AbstractArray{T, 1}
    data::Vector{T}
end
Base.size(A::UnfillableArray) = size(A.data)
Base.getindex(A::UnfillableArray, i::Int) = A.data[i]
Base.similar(A::UnfillableArray{T}) where {T} = UnfillableArray{T}(similar(A.data))
ArrayInterface.can_setindex(::Type{<:UnfillableArray}) = false

@testset "a buffer that cannot be zeroed errors instead of holding garbage" begin
    # The `fill!` is deliberately unguarded. Skipping it for non-setindexable buffers (which is
    # what the old `can_setindex` guard did) means silently returning uninitialized memory —
    # the failure mode is a wrong Jacobian far downstream. A new array type that cannot be
    # `fill!`ed should announce itself here and get an `init_similar_array!!` method, the way
    # the SparseArrays extension does.
    A = UnfillableArray([1.0, 2.0, 3.0])
    @test_throws Base.CanonicalIndexError Utils.safe_similar(A)
end

@testset "pass-through cases" begin
    # Non-numeric eltypes have no zero to fill with; they fall through untouched.
    y = Utils.safe_similar(["a", "b"])
    @test y isa Vector{String}
    @test length(y) == 2

    # Operators are returned as-is: they are not buffers, and `similar` on them is meaningless.
    op = MatrixOperator(rand(3, 3))
    @test Utils.safe_similar(op) === op
    @test Utils.safe_similar(op, Float32) === op
end
