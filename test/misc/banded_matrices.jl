# Miscellaneous Tests
using BandedMatrices, LinearAlgebra, NonlinearSolve, SparseArrays, XUnit

b = BandedMatrix(Ones(5, 5), (1, 1))
d = Diagonal(ones(5, 5))

@testcase "BandedMatrix vcat" begin
    @test NonlinearSolve._vcat(b, d) == vcat(sparse(b), d)
end
