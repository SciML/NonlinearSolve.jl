"""
    SteepestDescent()

Compute the descent direction as ``δu = -Jᵀfu``.

See also [`Dogleg`](@ref), [`NewtonDescent`](@ref), [`DampedNewtonDescent`](@ref).
"""
struct SteepestDescent <: AbstractDescentAlgorithm end

@concrete mutable struct SteepestDescentCache{pre_inverted} <: AbstractDescentCache
    δu
end

supports_line_search(::SteepestDescent) = true

@inline function init_cache(prob::AbstractNonlinearProblem, alg::SteepestDescent, J, fu,
        u; pre_inverted::Val{INV} = False, kwargs...) where {INV}
    @warn "Setting `pre_inverted = Val(true)` for `SteepestDescent` is not recommended." maxlog=1
    @bb δu = similar(u)
    return SteepestDescentCache{INV}(_restructure(u, δu))
end

@inline function SciMLBase.solve!(cache::SteepestDescentCache{INV}, J, fu;
        kwargs...) where {INV}
    J_ = INV ? inv(J) : J
    @bb cache.δu = transpose(J_) × vec(fu)
    @bb @. cache.δu *= -1
    return cache.δu
end
