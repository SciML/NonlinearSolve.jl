"""
    DiagonalStructure()

Preserves only the Diagonal of the Matrix.
"""
struct DiagonalStructure <: AbstractApproximateJacobianStructure end

NonlinearSolveBase.get_full_jacobian(cache, ::DiagonalStructure, J::Number) = J
function NonlinearSolveBase.get_full_jacobian(cache, ::DiagonalStructure, J)
    return Diagonal(Utils.safe_vec(J))
end

function (::DiagonalStructure)(J::AbstractMatrix; alias::Bool = false)
    @assert size(J, 1) == size(J, 2) "Diagonal Jacobian Structure must be square!"
    return LinearAlgebra.diag(J)
end
(::DiagonalStructure)(J::AbstractVector; alias::Bool = false) = alias ? J : @bb(copy(J))
(::DiagonalStructure)(J::Number; alias::Bool = false) = J

(::DiagonalStructure)(::Number, J_new::Number) = J_new
function (::DiagonalStructure)(J::AbstractVector, J_new::AbstractMatrix)
    if ArrayInterface.can_setindex(J)
        if ArrayInterface.fast_scalar_indexing(J)
            @simd ivdep for i in eachindex(J)
                @inbounds J[i] = J_new[i, i]
            end
        else
            J .= @view(J_new[diagind(J_new)])
        end
        return J
    end
    return LinearAlgebra.diag(J_new)
end
function (st::DiagonalStructure)(J::AbstractArray, J_new::AbstractMatrix)
    return ArrayInterface.restructure(J, st(vec(J), J_new))
end

"""
    FullStructure()

Stores the full matrix.
"""
struct FullStructure <: AbstractApproximateJacobianStructure end

NonlinearSolveBase.stores_full_jacobian(::FullStructure) = true

(::FullStructure)(J; alias::Bool = false) = alias ? J : @bb(copy(J))

function (::FullStructure)(J, J_new)
    J === J_new && return J
    @bb copyto!(J, J_new)
    return J
end
