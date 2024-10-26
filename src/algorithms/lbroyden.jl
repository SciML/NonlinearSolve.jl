"""
    LimitedMemoryBroyden(; max_resets::Int = 3, linesearch = nothing,
        threshold::Val = Val(10), reset_tolerance = nothing, alpha = nothing)

An implementation of `LimitedMemoryBroyden` [ziani2008autoadaptative](@cite) with resetting
and line search.

### Keyword Arguments

  - `max_resets`: the maximum number of resets to perform. Defaults to `3`.
  - `reset_tolerance`: the tolerance for the reset check. Defaults to
    `sqrt(eps(real(eltype(u))))`.
  - `threshold`: the number of vectors to store in the low rank approximation. Defaults
    to `Val(10)`.
  - `alpha`: The initial Jacobian inverse is set to be `(αI)⁻¹`. Defaults to `nothing`
    which implies `α = max(norm(u), 1) / (2 * norm(fu))`.
"""
function LimitedMemoryBroyden(; max_resets::Int = 3, linesearch = nothing,
        threshold::Union{Val, Int} = Val(10), reset_tolerance = nothing, alpha = nothing)
    threshold isa Int && (threshold = Val(threshold))
    initialization = BroydenLowRankInitialization(alpha, threshold)
    return ApproximateJacobianSolveAlgorithm{false, :LimitedMemoryBroyden}(; linesearch,
        descent = NewtonDescent(), update_rule = GoodBroydenUpdateRule(),
        max_resets, initialization, reinit_rule = NoChangeInStateReset(; reset_tolerance))
end

"""
    BroydenLowRankInitialization(alpha, threshold::Val)

An initialization for `LimitedMemoryBroyden` that uses a low rank approximation of the
Jacobian. The low rank updates to the Jacobian matrix corresponds to what SciPy calls
["simple"](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.broyden2.html#scipy-optimize-broyden2).
"""
@concrete struct BroydenLowRankInitialization <: AbstractJacobianInitialization
    alpha
    threshold <: Val
end

jacobian_initialized_preinverted(::BroydenLowRankInitialization) = true

function __internal_init(
        prob::AbstractNonlinearProblem, alg::BroydenLowRankInitialization,
        solver, f::F, fu, u, p; maxiters = 1000,
        internalnorm::IN = L2_NORM, kwargs...) where {F, IN}
    if u isa Number # Use the standard broyden
        return __internal_init(prob, IdentityInitialization(true, FullStructure()),
            solver, f, fu, u, p; maxiters, kwargs...)
    end
    # Pay to cost of slightly more allocations to prevent type-instability for StaticArrays
    α = inv(__initial_alpha(alg.alpha, u, fu, internalnorm))
    if u isa StaticArray
        J = BroydenLowRankJacobian(fu, u; alg.threshold, alpha = α)
    else
        threshold = min(_unwrap_val(alg.threshold), maxiters)
        J = BroydenLowRankJacobian(fu, u; threshold, alpha = α)
    end
    return InitializedApproximateJacobianCache(
        J, FullStructure(), alg, nothing, true, internalnorm)
end

function (cache::InitializedApproximateJacobianCache)(
        alg::BroydenLowRankInitialization, fu, u)
    α = __initial_alpha(alg.alpha, u, fu, cache.internalnorm)
    cache.J.idx = 0
    cache.J.alpha = inv(α)
    return
end

"""
    BroydenLowRankJacobian{T}(U, Vᵀ, idx, cache, alpha)

Low Rank Approximation of the Jacobian Matrix. Currently only used for
[`LimitedMemoryBroyden`](@ref). This computes the Jacobian as ``U \\times V^T``.
"""
@concrete mutable struct BroydenLowRankJacobian{T} <: AbstractSciMLOperator{T}
    U
    Vᵀ
    idx::Int
    cache
    alpha
end

__safe_inv!!(workspace, op::BroydenLowRankJacobian) = op  # Already Inverted form

@inline function __get_components(op::BroydenLowRankJacobian)
    op.idx ≥ size(op.U, 2) && return op.cache, op.U, transpose(op.Vᵀ)
    _cache = op.cache === nothing ? op.cache : view(op.cache, 1:(op.idx))
    return (_cache, view(op.U, :, 1:(op.idx)), transpose(view(op.Vᵀ, :, 1:(op.idx))))
end

Base.size(op::BroydenLowRankJacobian) = size(op.U, 1), size(op.Vᵀ, 1)
function Base.size(op::BroydenLowRankJacobian, d::Integer)
    return ifelse(d == 1, size(op.U, 1), size(op.Vᵀ, 1))
end

for op in (:adjoint, :transpose)
    # FIXME: adjoint might be a problem here. Fix if a complex number issue shows up
    @eval function Base.$(op)(operator::BroydenLowRankJacobian{T}) where {T}
        return BroydenLowRankJacobian{T}(
            operator.Vᵀ, operator.U, operator.idx, operator.cache, operator.alpha)
    end
end

# Storing the transpose to ensure contiguous memory on splicing
function BroydenLowRankJacobian(
        fu::StaticArray{S2, T2}, u::StaticArray{S1, T1}; alpha = true,
        threshold::Val{Th} = Val(10)) where {S1, S2, T1, T2, Th}
    T = promote_type(T1, T2)
    fuSize, uSize = Size(fu), Size(u)
    U = MArray{Tuple{prod(fuSize), Th}, T}(undef)
    Vᵀ = MArray{Tuple{prod(uSize), Th}, T}(undef)
    return BroydenLowRankJacobian{T}(U, Vᵀ, 0, nothing, T(alpha))
end

function BroydenLowRankJacobian(fu, u; threshold::Int = 10, alpha = true)
    T = promote_type(eltype(u), eltype(fu))
    U = __similar(fu, T, length(fu), threshold)
    Vᵀ = __similar(u, T, length(u), threshold)
    cache = __similar(u, T, threshold)
    return BroydenLowRankJacobian{T}(U, Vᵀ, 0, cache, T(alpha))
end

function Base.:*(J::BroydenLowRankJacobian, x::AbstractVector)
    J.idx == 0 && return -x
    cache, U, Vᵀ = __get_components(J)
    return U * (Vᵀ * x) .- J.alpha .* x
end

function LinearAlgebra.mul!(y::AbstractVector, J::BroydenLowRankJacobian, x::AbstractVector)
    if J.idx == 0
        @. y = -J.alpha * x
        return y
    end
    cache, U, Vᵀ = __get_components(J)
    @bb cache = Vᵀ × x
    mul!(y, U, cache)
    @bb @. y -= J.alpha * x
    return y
end

function Base.:*(x::AbstractVector, J::BroydenLowRankJacobian)
    J.idx == 0 && return -x
    cache, U, Vᵀ = __get_components(J)
    return Vᵀ' * (U' * x) .- J.alpha .* x
end

function LinearAlgebra.mul!(y::AbstractVector, x::AbstractVector, J::BroydenLowRankJacobian)
    if J.idx == 0
        @. y = -J.alpha * x
        return y
    end
    cache, U, Vᵀ = __get_components(J)
    @bb cache = transpose(U) × x
    mul!(y, transpose(Vᵀ), cache)
    @bb @. y -= J.alpha * x
    return y
end

function LinearAlgebra.mul!(J::BroydenLowRankJacobian, u::AbstractArray,
        vᵀ::LinearAlgebra.AdjOrTransAbsVec, α::Bool, β::Bool)
    @assert α & β
    idx_update = mod1(J.idx + 1, size(J.U, 2))
    copyto!(@view(J.U[:, idx_update]), _vec(u))
    copyto!(@view(J.Vᵀ[:, idx_update]), _vec(vᵀ))
    J.idx += 1
    return J
end

ArrayInterface.restructure(::BroydenLowRankJacobian, J::BroydenLowRankJacobian) = J
