module NonlinearSolveBaseSparseArraysExt

using ArrayInterface: ArrayInterface
using LinearAlgebra: transpose!
using SparseArrays: AbstractSparseArray, AbstractSparseMatrix, AbstractSparseMatrixCSC,
    SparseMatrixCSC, nonzeros, nzrange, rowvals, sparse

using NonlinearSolveBase: NonlinearSolveBase, Utils

# =============================================================================
# SparseArrays-specific implementations for NonlinearSolveBase
# =============================================================================

Utils.is_extension_loaded(::Val{:SparseArrays}) = true

"""
    structural_sparse(x::AbstractMatrix)

`SparseMatrixCSC` carrying the STRUCTURAL nonzero pattern of `x`, with `one(eltype(x))`
stored at every structural entry. Whether plain `sparse(x)` preserves zero-valued
structural entries depends on which specialized constructor exists: the band-copying
ones do (`sparse(Tridiagonal(zeros(4), zeros(5), zeros(4)))` keeps all 13 entries), but
`sparse(Diagonal(zeros(5)))` has no stored entries at all — so a prototype whose values
happen to be zero can silently lose pattern entries that sparse AD coloring needs.
Going through `findstructralnz` makes the conversion value-independent for every type.
"""
function Utils.structural_sparse(x::AbstractMatrix)
    rows, cols = ArrayInterface.findstructralnz(x)
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
    init_similar_array!!(x::AbstractSparseArray{<:Number})

Zero a freshly `similar`ed sparse array through its stored values.

The generic method zeroes with `fill!` which is not supported by sparse GPU arrays, so
we `fill!` the `nonzero`s in this case.
"""
function Utils.init_similar_array!!(x::AbstractSparseArray{<:Number})
    nzs = nonzeros(x)
    ArrayInterface.ismutable(nzs) && fill!(nzs, zero(eltype(x)))
    return x
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
    linsolve_workspace(A::AbstractSparseMatrix)

Build the inverse-Jacobian workspace on a densified copy of `A`: the inverse (and the
identity RHS) are generically dense, and `similar` on a sparse matrix would give sparse
buffers that cannot hold them efficiently. `linsolve_identity!!` calls with sparse `A`
then go through the dense workspace's linear-solve cache, which `copyto!`s `A` into its
dense buffer before factorizing.
"""
function Utils.linsolve_workspace(A::AbstractSparseMatrix)
    dense_A = Matrix(A)
    workspace, _ = Utils.linsolve_workspace(dense_A)
    return workspace, dense_A
end

struct SparseNormalFormWorkspace{Tv, Ti}
    Jᵀ::SparseMatrixCSC{Tv, Ti}
    accumulator::Vector{Tv}
    filled::Vector{Bool}
end

function Utils.normal_form_workspace(J::SparseMatrixCSC)
    return SparseNormalFormWorkspace(
        copy(transpose(J)), zeros(eltype(J), size(J, 2)), fill(false, size(J, 2))
    )
end

"""
    normal_form_jacobian!!(JᵀJ::SparseMatrixCSC, J::SparseMatrixCSC, workspace)

Overwrite the stored values of `JᵀJ` with those of `transpose(J) * J`. The column products
are accumulated in the same order as SparseArrays' Gustavson `spmatmul`, so the values are
bitwise identical to the allocating product. If the sparsity pattern of `JᵀJ` is not the
structural pattern of `transpose(J) * J` (e.g. `J` changed pattern), this falls back to the
allocating product.
"""
function Utils.normal_form_jacobian!!(
        JᵀJ::SparseMatrixCSC{Tv, Ti}, J::SparseMatrixCSC{Tv, Ti},
        workspace::SparseNormalFormWorkspace{Tv, Ti}
    ) where {Tv, Ti}
    n = size(J, 2)
    if size(JᵀJ) == (n, n) && size(workspace.Jᵀ) == (n, size(J, 1))
        transpose!(workspace.Jᵀ, J)
        sparse_normal_form_values!(JᵀJ, J, workspace) && return JᵀJ
    end
    return Utils.normal_form_jacobian!!(JᵀJ, J, nothing)
end

function sparse_normal_form_values!(C, J, workspace)
    (; Jᵀ, accumulator, filled) = workspace
    rows_J, vals_J = rowvals(J), nonzeros(J)
    rows_Jᵀ, vals_Jᵀ = rowvals(Jᵀ), nonzeros(Jᵀ)
    rows_C, vals_C = rowvals(C), nonzeros(C)
    @inbounds for i in axes(J, 2)
        nfilled = 0
        for jp in nzrange(J, i)
            j, Jji = rows_J[jp], vals_J[jp]
            for kp in nzrange(Jᵀ, j)
                k = rows_Jᵀ[kp]
                v = vals_Jᵀ[kp] * Jji
                if filled[k]
                    accumulator[k] += v
                else
                    accumulator[k] = v
                    filled[k] = true
                    nfilled += 1
                end
            end
        end
        matched = nfilled == length(nzrange(C, i))
        for p in nzrange(C, i)
            k = rows_C[p]
            if filled[k]
                vals_C[p] = accumulator[k]
                filled[k] = false
            else
                matched = false
            end
        end
        if !matched
            fill!(filled, false)
            return false
        end
    end
    return true
end

end
