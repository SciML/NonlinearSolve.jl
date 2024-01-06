# Ignores NaN

_mutable_zero(x) = zero(x)
_mutable_zero(x::SArray) = MArray(x)

_mutable(x) = x
_mutable(x::SArray) = MArray(x)

# __maybe_mutable(x, ::AbstractFiniteDifferencesMode) = _mutable(x)
# The shadow allocated for Enzyme needs to be mutable
__maybe_mutable(x, ::AutoSparseEnzyme) = _mutable(x)
__maybe_mutable(x, _) = x


function __init_low_rank_jacobian(u::StaticArray{S1, T1}, fu::StaticArray{S2, T2},
        ::Val{threshold}) where {S1, S2, T1, T2, threshold}
    T = promote_type(T1, T2)
    fuSize, uSize = Size(fu), Size(u)
    Vᵀ = MArray{Tuple{threshold, prod(uSize)}, T}(undef)
    U = MArray{Tuple{prod(fuSize), threshold}, T}(undef)
    return U, Vᵀ
end
function __init_low_rank_jacobian(u, fu, ::Val{threshold}) where {threshold}
    Vᵀ = similar(u, threshold, length(u))
    U = similar(u, length(fu), threshold)
    return U, Vᵀ
end
