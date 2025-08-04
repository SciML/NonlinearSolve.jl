module NonlinearSolveBaseBandedMatricesExt

using BandedMatrices: BandedMatrix
using LinearAlgebra: Diagonal

using NonlinearSolveBase: NonlinearSolveBase, Utils

# This is used if we vcat a Banded Jacobian with a Diagonal Matrix in Levenberg
@inline function Utils.faster_vcat(B::BandedMatrix, D::Diagonal)
    if !Utils.is_extension_loaded(Val(:SparseArrays))
        @warn "Load `SparseArrays` for an optimized vcat for BandedMatrices."
        # Convert BandedMatrix to full Matrix for fallback vcat since direct vcat fails
        return vcat(Matrix(B), D)
    end
    return vcat(Utils.make_sparse(B), D)  # Use sparse conversion when available
end

end
