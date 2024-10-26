module NonlinearSolveBaseSparseArraysExt

using NonlinearSolveBase: NonlinearSolveBase, Utils
using SparseArrays: AbstractSparseMatrix, AbstractSparseMatrixCSC, nonzeros, sparse

function NonlinearSolveBase.NAN_CHECK(x::AbstractSparseMatrixCSC)
    return any(NonlinearSolveBase.NAN_CHECK, nonzeros(x))
end

NonlinearSolveBase.sparse_or_structured_prototype(::AbstractSparseMatrix) = true

Utils.maybe_symmetric(x::AbstractSparseMatrix) = x

Utils.make_sparse(x) = sparse(x)

end
