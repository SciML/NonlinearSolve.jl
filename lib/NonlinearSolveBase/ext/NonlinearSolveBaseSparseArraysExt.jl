module NonlinearSolveBaseSparseArraysExt

using NonlinearSolveBase: NonlinearSolveBase
using SparseArrays: AbstractSparseMatrix, AbstractSparseMatrixCSC, nonzeros

function NonlinearSolveBase.NAN_CHECK(x::AbstractSparseMatrixCSC)
    return any(NonlinearSolveBase.NAN_CHECK, nonzeros(x))
end

NonlinearSolveBase.sparse_or_structured_prototype(::AbstractSparseMatrix) = true

end
