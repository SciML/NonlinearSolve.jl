module NonlinearSolveBaseSparseArraysExt

using SparseArrays: AbstractSparseMatrix, AbstractSparseMatrixCSC, nonzeros, sparse

using NonlinearSolveBase: NonlinearSolveBase, Utils

function NonlinearSolveBase.NAN_CHECK(x::AbstractSparseMatrixCSC)
    return any(NonlinearSolveBase.NAN_CHECK, nonzeros(x))
end

NonlinearSolveBase.sparse_or_structured_prototype(::AbstractSparseMatrix) = true

Utils.maybe_symmetric(x::AbstractSparseMatrix) = x

Utils.make_sparse(x) = sparse(x)

Utils.condition_number(J::AbstractSparseMatrix) = Utils.condition_number(Matrix(J))

function Utils.maybe_pinv!!_workspace(A::AbstractSparseMatrix)
    dense_A = Matrix(A)
    return dense_A, copy(dense_A)
end

end
