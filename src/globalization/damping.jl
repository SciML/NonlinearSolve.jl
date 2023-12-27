abstract type AbstractJacobianDampingStrategy end

struct NoJacobianDamping <: AbstractJacobianDampingStrategy end

@inline (alg::NoJacobianDamping)(J; alias = true) = JacobianDampingCache(alg, J)

@concrete mutable struct JacobianDampingCache
    alg
    J
end

@inline (::JacobianDampingCache{<:NoJacobianDamping})(J) = J
@inline (::JacobianDampingCache{<:NoJacobianDamping})(J, ::Val{INV}) where {INV} = J
