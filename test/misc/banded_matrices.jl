# Miscellaneous Tests
using BandedMatrices, LinearAlgebra, NonlinearSolve, SparseArrays, Test

b = BandedMatrix(Ones(5, 5), (1, 1))
d = Diagonal(ones(5, 5))

@test NonlinearSolve._vcat(b, d) == vcat(sparse(b), d)
