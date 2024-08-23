module NonlinearSolveBaseSparseArraysExt

using NonlinearSolveBase: NonlinearSolveBase
using SparseArrays: AbstractSparseMatrixCSC, nonzeros

function NonlinearSolveBase.NAN_CHECK(x::AbstractSparseMatrixCSC)
    return any(NonlinearSolveBase.NAN_CHECK, nonzeros(x))
end

end