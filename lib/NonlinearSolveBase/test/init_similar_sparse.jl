using NonlinearSolveBase
using NonlinearSolveBase: Utils
using ArrayInterface
using SparseArrays
using Test

@testset "safe_similar zeroes sparse arrays" begin
    A = sprand(20, 20, 0.2)
    @test all(iszero, nonzeros(Utils.safe_similar(A)))

    v = sprand(20, 0.2)
    @test all(iszero, nonzeros(Utils.safe_similar(v)))
end

# GPU sparse matrices (CuSparseMatrixCSC, ...) implement no `setindex!` at all, so the
# `can_setindex`-guarded `fill!` in the generic method is skipped for them and they used to
# come back holding uninitialized memory. They are still mutable through `nonzeros`. Stand in
# for them here with a sparse matrix that behaves the same way, so this is covered off-GPU.
struct NoSetindexSparseMatrix{Tv, Ti} <: AbstractSparseMatrix{Tv, Ti}
    m::Int
    n::Int
    vals::Vector{Tv}
end
Base.size(A::NoSetindexSparseMatrix) = (A.m, A.n)
SparseArrays.nonzeros(A::NoSetindexSparseMatrix) = A.vals
Base.similar(A::NoSetindexSparseMatrix{Tv, Ti}) where {Tv, Ti} =
    NoSetindexSparseMatrix{Tv, Ti}(A.m, A.n, similar(A.vals))
ArrayInterface.can_setindex(::Type{<:NoSetindexSparseMatrix}) = false

@testset "safe_similar zeroes sparse arrays that cannot setindex!" begin
    A = NoSetindexSparseMatrix{Float64, Int}(4, 4, [1.0, 2.0, 3.0])
    @test !ArrayInterface.can_setindex(A)
    @test all(iszero, nonzeros(Utils.safe_similar(A)))
end
