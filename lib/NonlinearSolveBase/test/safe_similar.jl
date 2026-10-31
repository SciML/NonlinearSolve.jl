using NonlinearSolveBase
using NonlinearSolveBase: Utils
using ArrayInterface
using SparseArrays
using Test

@testset "dense buffers are zeroed" begin
    @test all(iszero, Utils.safe_similar(rand(4)))
    @test all(iszero, Utils.safe_similar(rand(4), 2, 3))
    # BigFloat is the original motivation: unfilled, these entries are undef *references* and
    # `iszero` on them throws rather than returning garbage.
    @test all(iszero, Utils.safe_similar(BigFloat[1, 2, 3]))
end

@testset "sparse buffers are zeroed through their stored values" begin
    A = sprand(20, 20, 0.2)
    B = Utils.safe_similar(A)
    @test all(iszero, nonzeros(B))
    # The structural pattern must survive: it is what the prototype exists to carry, and sparse
    # AD coloring decompresses into exactly these entries.
    @test nnz(B) == nnz(A)
    @test rowvals(B) == rowvals(A)
    @test SparseArrays.getcolptr(B) == SparseArrays.getcolptr(A)
    @test all(iszero, nonzeros(Utils.safe_similar(sprand(20, 0.2))))
end

# GPU sparse matrices (`CuSparseMatrixCSC`, ...) implement no `setindex!` at all, so the generic
# `fill!` is skipped for them, but they are mutable through `nonzeros` — which is all the zeroing
# needs. Stand in for them with a sparse type that behaves the same way, so this is covered
# off-GPU. Plain `SparseMatrixCSC` would not: its `fill!` already goes through `nonzeros`.
struct NoSetindexSparseMatrix{Tv, Ti} <: AbstractSparseMatrix{Tv, Ti}
    m::Int
    n::Int
    vals::Vector{Tv}
end
Base.size(A::NoSetindexSparseMatrix) = (A.m, A.n)
SparseArrays.nonzeros(A::NoSetindexSparseMatrix) = A.vals
function Base.similar(A::NoSetindexSparseMatrix{Tv, Ti}) where {Tv, Ti}
    # NaN stands in for uninitialized memory, so a skipped zeroing fails the test reliably.
    return NoSetindexSparseMatrix{Tv, Ti}(A.m, A.n, fill!(similar(A.vals), NaN))
end
ArrayInterface.can_setindex(::Type{<:NoSetindexSparseMatrix}) = false

@testset "sparse buffers that cannot `setindex!` are still zeroed" begin
    A = NoSetindexSparseMatrix{Float64, Int}(4, 4, [1.0, 2.0, 3.0])
    @test !ArrayInterface.can_setindex(A)
    @test all(iszero, nonzeros(Utils.safe_similar(A)))
end
