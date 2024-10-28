"""
    InitializedApproximateJacobianCache(J, structure, alg, cache, initialized::Bool,
        internalnorm)

A cache for Approximate Jacobian.

### Arguments

  - `J`: The current Jacobian.
  - `structure`: The structure of the Jacobian.
  - `alg`: The initialization algorithm.
  - `cache`: The Jacobian cache [`NonlinearSolve.JacobianCache`](@ref) (if needed).
  - `initialized`: A boolean indicating whether the Jacobian has been initialized.
  - `internalnorm`: The norm to be used.

### Interface

```julia
(cache::InitializedApproximateJacobianCache)(::Nothing)
```

Returns the current Jacobian `cache.J` with the proper `structure`.

```julia
__internal_solve!(cache::InitializedApproximateJacobianCache, fu, u, ::Val{reinit})
```

Solves for the Jacobian `cache.J` and returns it. If `reinit` is `true`, then the Jacobian
is reinitialized.
"""
@concrete mutable struct InitializedApproximateJacobianCache <: AbstractJacobianCache
    J
    structure
    alg
    cache
    initialized::Bool
    internalnorm
end

function InternalAPI.reinit!(cache::InitializedApproximateJacobianCache; kwargs...)
    cache.initialized = false
end

# XXX: Implement
# @internal_caches InitializedApproximateJacobianCache :cache

function (cache::InitializedApproximateJacobianCache)(::Nothing)
    return NonlinearSolveBase.get_full_jacobian(cache, cache.structure, cache.J)
end

function InternalAPI.solve!(
        cache::InitializedApproximateJacobianCache, fu, u, reinit::Val
)
    if reinit isa Val{true} || !cache.initialized
        cache(cache.alg, fu, u)
        cache.initialized = true
    end
    if NonlinearSolveBase.stores_full_jacobian(cache.structure)
        full_J = cache.J
    else
        full_J = NonlinearSolveBase.get_full_jacobian(cache, cache.structure, cache.J)
    end
    return full_J
end

"""
    IdentityInitialization(alpha, structure)

Initialize the Jacobian to be an Identity Matrix scaled by `alpha` and maintain the
structure as specified by `structure`.
"""
@concrete struct IdentityInitialization <: AbstractJacobianInitialization
    alpha
    structure
end

function InternalAPI.init(
        prob::AbstractNonlinearProblem, alg::IdentityInitialization, solver, f::F,
        fu, u, p; internalnorm::IN = L2_NORM, kwargs...
) where {F, IN}
    α = Utils.initial_jacobian_scaling_alpha(alg.alpha, u, fu, internalnorm)
    if u isa Number
        J = α
    else
        if alg.structure isa DiagonalStructure
            @assert length(u)==length(fu) "Diagonal Jacobian Structure must be square!"
            J = one.(Utils.safe_vec(fu)) .* α
        else
            # A simple trick to get the correct jacobian structure
            J = alg.structure(Utils.make_identity!!(vec(fu) * vec(u)', α); alias = true)
        end
    end
    return InitializedApproximateJacobianCache(
        J, alg.structure, alg, nothing, true, internalnorm
    )
end

function (cache::InitializedApproximateJacobianCache)(
        alg::IdentityInitialization, fu, u
)
    α = Utils.initial_jacobian_scaling_alpha(alg.alpha, u, fu, cache.internalnorm)
    cache.J = Utils.make_identity!!(cache.J, α)
    return
end

"""
    TrueJacobianInitialization(structure, autodiff)

Initialize the Jacobian to be the true Jacobian and maintain the structure as specified
by `structure`. `autodiff` is used to compute the true Jacobian and if not specified we
make a selection automatically.
"""
@concrete struct TrueJacobianInitialization <: AbstractJacobianInitialization
    structure
    autodiff
end

function InternalAPI.init(
        prob::AbstractNonlinearProblem, alg::TrueJacobianInitialization,
        solver, f::F, fu, u, p; stats, linsolve = missing,
        internalnorm::IN = L2_NORM, kwargs...
) where {F, IN}
    autodiff = NonlinearSolveBase.select_jacobian_autodiff(prob, alg.autodiff)
    jac_cache = NonlinearSolveBase.construct_jacobian_cache(
        prob, solver, prob.f, fu, u, p; stats, autodiff, linsolve
    )
    J = alg.structure(jac_cache(nothing))
    return InitializedApproximateJacobianCache(
        J, alg.structure, alg, jac_cache, false, internalnorm
    )
end

function (cache::InitializedApproximateJacobianCache)(::TrueJacobianInitialization, fu, u)
    cache.J = cache.structure(cache.J, cache.cache(u))
    return
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

NonlinearSolveBase.jacobian_initialized_preinverted(::BroydenLowRankInitialization) = true

function InternalAPI.init(
        prob::AbstractNonlinearProblem, alg::BroydenLowRankInitialization,
        solver, f::F, fu, u, p; internalnorm::IN = L2_NORM, kwargs...
) where {F, IN}
    if u isa Number # Use the standard broyden
        return InternalAPI.init(
            prob, IdentityInitialization(true, FullStructure()),
            solver, f, fu, u, p; internalnorm, kwargs...
        )
    end
    # Pay to cost of slightly more allocations to prevent type-instability for StaticArrays
    α = inv(Utils.initial_jacobian_scaling_alpha(alg.alpha, u, fu, internalnorm))
    if u isa StaticArray
        J = BroydenLowRankJacobian(fu, u; alg.threshold, alpha = α)
    else
        threshold = min(Utils.unwrap_val(alg.threshold), maxiters)
        J = BroydenLowRankJacobian(fu, u; threshold, alpha = α)
    end
    return InitializedApproximateJacobianCache(
        J, FullStructure(), alg, nothing, true, internalnorm
    )
end

function (cache::InitializedApproximateJacobianCache)(
        alg::BroydenLowRankInitialization, fu, u
)
    α = Utils.initial_jacobian_scaling_alpha(alg.alpha, u, fu, cache.internalnorm)
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

Utils.maybe_pinv!!(workspace, A::BroydenLowRankJacobian) = A  # Already Inverted form

function get_components(op::BroydenLowRankJacobian)
    op.idx ≥ size(op.U, 2) && return op.cache, op.U, transpose(op.Vᵀ)
    cache = op.cache === nothing ? op.cache : view(op.cache, 1:(op.idx))
    return cache, view(op.U, :, 1:(op.idx)), transpose(view(op.Vᵀ, :, 1:(op.idx)))
end

Base.size(op::BroydenLowRankJacobian) = size(op.U, 1), size(op.Vᵀ, 1)

function Base.transpose(op::BroydenLowRankJacobian{T}) where {T}
    return BroydenLowRankJacobian{T}(op.Vᵀ, op.U, op.idx, op.cache, op.alpha)
end
Base.adjoint(op::BroydenLowRankJacobian{<:Real}) = transpose(op)

# Storing the transpose to ensure contiguous memory on splicing
function BroydenLowRankJacobian(
    fu::StaticArray, u::StaticArray; alpha = true, threshold::Val = Val(10)
)
    T = promote_type(eltype(u), eltype(fu))
    U = MArray{Tuple{prod(Size(fu)), Utils.unwrap_val(threshold)}, T}(undef)
    Vᵀ = MArray{Tuple{prod(Size(u)), Utils.unwrap_val(threshold)}, T}(undef)
    return BroydenLowRankJacobian{T}(U, Vᵀ, 0, nothing, T(alpha))
end

function BroydenLowRankJacobian(fu, u; threshold::Int = 10, alpha = true)
    T = promote_type(eltype(u), eltype(fu))
    U = Utils.safe_similar(fu, T, length(fu), threshold)
    Vᵀ = Utils.safe_similar(u, T, length(u), threshold)
    cache = Utils.safe_similar(u, T, threshold)
    return BroydenLowRankJacobian{T}(U, Vᵀ, 0, cache, T(alpha))
end

function Base.:*(J::BroydenLowRankJacobian, x::AbstractVector)
    J.idx == 0 && return -x
    _, U, Vᵀ = get_components(J)
    return U * (Vᵀ * x) .- J.alpha .* x
end

function LinearAlgebra.mul!(y::AbstractVector, J::BroydenLowRankJacobian, x::AbstractVector)
    if J.idx == 0
        @. y = -J.alpha * x
        return y
    end
    _, U, Vᵀ = get_components(J)
    @bb cache = Vᵀ × x
    mul!(y, U, cache)
    @bb @. y -= J.alpha * x
    return y
end

function Base.:*(x::AbstractVector, J::BroydenLowRankJacobian)
    J.idx == 0 && return -x
    _, U, Vᵀ = get_components(J)
    return Vᵀ' * (U' * x) .- J.alpha .* x
end

function LinearAlgebra.mul!(y::AbstractVector, x::AbstractVector, J::BroydenLowRankJacobian)
    if J.idx == 0
        @. y = -J.alpha * x
        return y
    end
    _, U, Vᵀ = get_components(J)
    @bb cache = transpose(U) × x
    mul!(y, transpose(Vᵀ), cache)
    @bb @. y -= J.alpha * x
    return y
end

function LinearAlgebra.mul!(
    J::BroydenLowRankJacobian, u::AbstractArray, vᵀ::LinearAlgebra.AdjOrTransAbsVec,
    α::Bool, β::Bool
)
    @assert α & β
    idx_update = mod1(J.idx + 1, size(J.U, 2))
    copyto!(@view(J.U[:, idx_update]), Utils.safe_vec(u))
    copyto!(@view(J.Vᵀ[:, idx_update]), Utils.safe_vec(vᵀ))
    J.idx += 1
    return J
end

ArrayInterface.restructure(::BroydenLowRankJacobian, J::BroydenLowRankJacobian) = J
