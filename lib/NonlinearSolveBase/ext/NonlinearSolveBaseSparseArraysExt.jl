module NonlinearSolveBaseSparseArraysExt

using SparseArrays: AbstractSparseMatrix, AbstractSparseMatrixCSC, nonzeros, sparse

using NonlinearSolveBase: NonlinearSolveBase, Utils

# =============================================================================
# SparseArrays-specific implementations for NonlinearSolveBase
# =============================================================================

Utils.is_extension_loaded(::Val{:SparseArrays}) = true

"""
    structural_sparse(x::AbstractMatrix)

`SparseMatrixCSC` carrying the STRUCTURAL nonzero pattern of `x`, with `one(eltype(x))`
stored at every structural entry. Unlike `sparse(x)`, which keeps only value-nonzero
entries, this preserves structural entries whose values happen to be zero — a
`Tridiagonal` prototype with zero-valued bands must not lose its bands when converted
for pattern purposes.
"""
function Utils.structural_sparse(x::AbstractMatrix)
    rows, cols = NonlinearSolveBase.ArrayInterface.findstructralnz(x)
    # indexed comprehensions, not collect: the structured-matrix index iterators
    # (e.g. ArrayInterface.TridiagonalIndex) advertise eltype Any and fail collect
    I = Int[rows[k] for k in 1:length(rows)]
    J = Int[cols[k] for k in 1:length(cols)]
    return sparse(I, J, fill(one(eltype(x)), length(I)), size(x, 1), size(x, 2))
end

Utils.structural_sparse(x::AbstractSparseMatrixCSC) = x

"""
    NAN_CHECK(x::AbstractSparseMatrixCSC)

Efficient NaN checking for sparse matrices that only checks nonzero entries.
This is more efficient than checking all entries including structural zeros.
"""
function NonlinearSolveBase.NAN_CHECK(x::AbstractSparseMatrixCSC)
    return any(NonlinearSolveBase.NAN_CHECK, nonzeros(x))
end

"""
    sparse_or_structured_prototype(::AbstractSparseMatrix)

Indicates that AbstractSparseMatrix types are considered sparse/structured.
This enables sparse automatic differentiation pathways.
"""
NonlinearSolveBase.sparse_or_structured_prototype(::AbstractSparseMatrix) = true

"""
    maybe_symmetric(x::AbstractSparseMatrix)

For sparse matrices, return as-is without wrapping in Symmetric.
Sparse matrices handle symmetry more efficiently without wrappers.
"""
Utils.maybe_symmetric(x::AbstractSparseMatrix) = x

"""
    make_sparse(x)

Convert a matrix to sparse format using SparseArrays.sparse().
Used primarily in BandedMatrices extension for efficient concatenation.
"""
Utils.make_sparse(x) = sparse(x)

"""
    condition_number(J::AbstractSparseMatrix)

Compute condition number of sparse matrix by converting to dense.
This is necessary because efficient sparse condition number computation
is not generally available.
"""
Utils.condition_number(J::AbstractSparseMatrix) = Utils.condition_number(Matrix(J))

"""
    maybe_pinv!!_workspace(A::AbstractSparseMatrix)

Prepare workspace for pseudo-inverse computation of sparse matrices.
Converts to dense format since sparse pseudo-inverse is not efficient.
Returns (dense_A, copy(dense_A)) for in-place operations.
"""
function Utils.maybe_pinv!!_workspace(A::AbstractSparseMatrix)
    dense_A = Matrix(A)
    return dense_A, copy(dense_A)
end

end
