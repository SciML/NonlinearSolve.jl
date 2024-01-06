"""
    LimitedMemoryBroyden(; max_resets::Int = 3, linesearch = nothing,
        threshold::Int = 10, reset_tolerance = nothing)

An implementation of `LimitedMemoryBroyden` with resetting and line search.

### Keyword Arguments

  - `max_resets`: the maximum number of resets to perform. Defaults to `3`.
  - `reset_tolerance`: the tolerance for the reset check. Defaults to
    `sqrt(eps(real(eltype(u))))`.
  - `threshold`: the number of vectors to store in the low rank approximation. Defaults
    to `10`.

### References

[1] van de Rotten, Bart, and Sjoerd Verduyn Lunel. "A limited memory Broyden method to solve
high-dimensional systems of nonlinear equations." EQUADIFF 2003. 2005. 196-201.
"""
function LimitedMemoryBroyden(; max_resets::Int = 3, linesearch = NoLineSearch(),
        threshold::Union{Val, Int} = 10, reset_tolerance = nothing)
    if threshold isa Val
        Base.depwarn("Passing `Val(threshold)` is deprecated. Use `threshold` instead.",
            :LimitedMemoryBroyden)
    end
    return ApproximateJacobianSolveAlgorithm{false, :LimitedMemoryBroyden}(; linesearch,
        descent = NewtonDescent(), update_rule = GoodBroydenUpdateRule(), max_resets,
        initialization = BroydenLowRankInitialization(_unwrap_val(threshold)),
        reinit_rule = NoChangeInStateReset(; reset_tolerance))
end

struct BroydenLowRankInitialization <: AbstractJacobianInitialization
    threshold::Int
end

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::BroydenLowRankInitialization,
        solver, f::F, fu, u, p; maxiters = 1000, kwargs...) where {F}
    threshold = min(alg.threshold, maxiters)
    J = BroydenLowRankJacobian(fu, u; threshold)
    return InitializedApproximateJacobianCache(J, FullStructure(), alg, nothing, true, 0.0)
end

function (cache::InitializedApproximateJacobianCache)(alg::BroydenLowRankInitialization, u)
    cache.J.idx = 0
    return
end

@concrete mutable struct BroydenLowRankJacobian{T} <: AbstractNonlinearSolveOperator{T}
    U
    Vᵀ
    idx::Int
    cache
end

__safe_inv!!(workspace, op::BroydenLowRankJacobian) = op  # Already Inverted form

@inline function __get_components(op::BroydenLowRankJacobian)
    op.idx ≥ size(op.U, 2) && return op.cache, op.U, op.Vᵀ
    return view(op.cache, 1:(op.idx)), view(op.U, :, 1:(op.idx)), view(op.Vᵀ, 1:(op.idx), :)
end

Base.size(op::BroydenLowRankJacobian) = size(op.U, 1), size(op.Vᵀ, 2)
function Base.size(op::BroydenLowRankJacobian, d::Integer)
    return ifelse(d == 1, size(op.U, 1), size(op.Vᵀ, 2))
end

for op in (:adjoint, :transpose)
    @eval function Base.$(op)(operator::BroydenLowRankJacobian{T}) where {T}
        return BroydenLowRankJacobian{T}($(op)(operator.Vᵀ), $(op)(operator.U),
            operator.idx, operator.cache)
    end
end

function BroydenLowRankJacobian(fu, u; threshold = 10)
    T = promote_type(eltype(u), eltype(fu))
    # TODO: Mutable for StaticArrays
    U = similar(fu, T, length(fu), threshold)
    Vᵀ = similar(u, T, threshold, length(u))
    cache = similar(u, T, threshold)
    return BroydenLowRankJacobian{T}(U, Vᵀ, 0, cache)
end

function LinearAlgebra.mul!(y::AbstractVector, J::BroydenLowRankJacobian, x::AbstractVector)
    if J.idx == 0
        @. y = -x
        return y
    end
    cache, U, Vᵀ = __get_components(J)
    @bb cache = Vᵀ × x
    mul!(y, U, cache)
    @bb @. y -= x
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, x::AbstractVector, J::BroydenLowRankJacobian)
    if J.idx == 0
        @. y = -x
        return y
    end
    cache, U, Vᵀ = __get_components(J)
    @bb cache = transpose(U) × x
    mul!(y, transpose(Vᵀ), cache)
    @bb @. y -= x
    return y
end

function LinearAlgebra.mul!(J::BroydenLowRankJacobian, u,
        vᵀ::LinearAlgebra.AdjOrTransAbsVec, α::Bool, β::Bool)
    @assert α & β
    idx_update = mod1(J.idx + 1, size(J.U, 2))
    copyto!(@view(J.U[:, idx_update]), _vec(u))
    copyto!(@view(J.Vᵀ[idx_update, :]), _vec(vᵀ))
    J.idx += 1
    return J
end

restructure(::BroydenLowRankJacobian, J::BroydenLowRankJacobian) = J
