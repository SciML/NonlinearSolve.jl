_batch_transpose(x) = reshape(x, 1, size(x)...)

_batched_mul(x, y) = x * y

function _batched_mul(x::AbstractArray{T, 3}, y::AbstractMatrix) where {T}
    return dropdims(batched_mul(x, reshape(y, size(y, 1), 1, size(y, 2))); dims = 2)
end

function _batched_mul(x::AbstractMatrix, y::AbstractArray{T, 3}) where {T}
    return batched_mul(reshape(x, size(x, 1), 1, size(x, 2)), y)
end

function _batched_mul(x::AbstractArray{T1, 3}, y::AbstractArray{T2, 3}) where {T1, T2}
    return batched_mul(x, y)
end

function _init_J_batched(x::AbstractMatrix{T}) where {T}
    J = ArrayInterfaceCore.zeromatrix(x[:, 1])
    if ismutable(x)
        J[diagind(J)] .= one(eltype(x))
    else
        J += I
    end
    return repeat(J, 1, 1, size(x, 2))
end
