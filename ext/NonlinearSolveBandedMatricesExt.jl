module NonlinearSolveBandedMatricesExt

using BandedMatrices: BandedMatrix
using LinearAlgebra: Diagonal
using NonlinearSolve: NonlinearSolve
using SparseArrays: sparse

# This is used if we vcat a Banded Jacobian with a Diagonal Matrix in Levenberg
@inline NonlinearSolve._vcat(B::BandedMatrix, D::Diagonal) = vcat(sparse(B), D)

end
