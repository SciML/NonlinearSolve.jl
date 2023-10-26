module NonlinearSolveBandedMatricesExt

using BandedMatrices, LinearAlgebra, NonlinearSolve, SparseArrays

# This is used if we vcat a Banded Jacobian with a Diagonal Matrix in Levenberg
@inline NonlinearSolve._vcat(B::BandedMatrix, D::Diagonal) = vcat(sparse(B), D)

end
