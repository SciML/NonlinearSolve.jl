# Jacobian Structure
"""
    DiagonalStructure()

Preserves only the Diagonal of the Matrix.
"""
struct DiagonalStructure <: AbstractApproximateJacobianStructure end

get_full_jacobian(cache, ::DiagonalStructure, J::Number) = J
get_full_jacobian(cache, ::DiagonalStructure, J) = Diagonal(_vec(J))

function (::DiagonalStructure)(J::AbstractMatrix; alias::Bool = false)
    @assert size(J, 1)==size(J, 2) "Diagonal Jacobian Structure must be square!"
    return diag(J)
end
(::DiagonalStructure)(J::AbstractVector; alias::Bool = false) = alias ? J : @bb(copy(J))
(::DiagonalStructure)(J::Number; alias::Bool = false) = J

(::DiagonalStructure)(::Number, J_new::Number) = J_new
function (::DiagonalStructure)(J::AbstractVector, J_new::AbstractMatrix)
    if __can_setindex(J)
        if fast_scalar_indexing(J)
            @inbounds for i in eachindex(J)
                J[i] = J_new[i, i]
            end
        else
            @.. broadcast=false J=@view(J_new[diagind(J_new)])
        end
        return J
    end
    return diag(J_new)
end
function (st::DiagonalStructure)(J::AbstractArray, J_new::AbstractMatrix)
    return _restructure(J, st(vec(J), J_new))
end

"""
    FullStructure()

Stores the full matrix.
"""
struct FullStructure <: AbstractApproximateJacobianStructure end

stores_full_jacobian(::FullStructure) = true

(::FullStructure)(J; alias::Bool = false) = alias ? J : @bb(copy(J))

function (::FullStructure)(J, J_new)
    J === J_new && return J
    @bb copyto!(J, J_new)
    return J
end

# Initialization Strategies
"""
    IdentityInitialization(alpha, structure)

Initialize the Jacobian to be an Identity Matrix scaled by `alpha` and maintain the
structure as specified by `structure`.
"""
@concrete struct IdentityInitialization <: AbstractJacobianInitialization
    alpha
    structure
end

function __internal_init(
        prob::AbstractNonlinearProblem, alg::IdentityInitialization, solver, f::F,
        fu, u::Number, p; internalnorm::IN = DEFAULT_NORM, kwargs...) where {F, IN}
    α = __initial_alpha(alg.alpha, u, fu, internalnorm)
    return InitializedApproximateJacobianCache(
        α, alg.structure, alg, nothing, true, internalnorm)
end
function __internal_init(prob::AbstractNonlinearProblem, alg::IdentityInitialization,
        solver, f::F, fu::StaticArray, u::StaticArray, p;
        internalnorm::IN = DEFAULT_NORM, kwargs...) where {IN, F}
    α = __initial_alpha(alg.alpha, u, fu, internalnorm)
    if alg.structure isa DiagonalStructure
        @assert length(u)==length(fu) "Diagonal Jacobian Structure must be square!"
        J = one.(_vec(fu)) .* α
    else
        T = promote_type(eltype(u), eltype(fu))
        if fu isa SArray
            J_ = SArray{Tuple{prod(Size(fu)), prod(Size(u))}, T}(I * α)
        else
            J_ = MArray{Tuple{prod(Size(fu)), prod(Size(u))}, T}(I * α)
        end
        J = alg.structure(J_; alias = true)
    end
    return InitializedApproximateJacobianCache(
        J, alg.structure, alg, nothing, true, internalnorm)
end
function __internal_init(
        prob::AbstractNonlinearProblem, alg::IdentityInitialization, solver,
        f::F, fu, u, p; internalnorm::IN = DEFAULT_NORM, kwargs...) where {F, IN}
    α = __initial_alpha(alg.alpha, u, fu, internalnorm)
    if alg.structure isa DiagonalStructure
        @assert length(u)==length(fu) "Diagonal Jacobian Structure must be square!"
        J = one.(_vec(fu)) .* α
    else
        J_ = similar(fu, promote_type(eltype(fu), eltype(u)), length(fu), length(u))
        J = alg.structure(__make_identity!!(J_, α); alias = true)
    end
    return InitializedApproximateJacobianCache(
        J, alg.structure, alg, nothing, true, internalnorm)
end

@inline function __initial_alpha(α, u, fu, internalnorm::F) where {F}
    return convert(promote_type(eltype(u), eltype(fu)), α)
end
@inline function __initial_alpha(::Nothing, u, fu, internalnorm::F) where {F}
    fu_norm = internalnorm(fu)
    return ifelse(fu_norm ≥ 1e-5, (2 * fu_norm) / max(norm(u), true),
        __initial_alpha(true, u, fu, internalnorm))
end

@inline __make_identity!!(A::Number, α) = one(A) * α
@inline __make_identity!!(A::AbstractVector, α) = __can_setindex(A) ? (A .= α) :
                                                  (one.(A) .* α)
@inline function __make_identity!!(A::AbstractMatrix{T}, α) where {T}
    if A isa SMatrix
        Sz = Size(A)
        return SArray{Tuple{Sz[1], Sz[2]}, eltype(Sz)}(I * α)
    end
    @assert __can_setindex(A) "__make_identity!!(::AbstractMatrix) only works on mutable arrays!"
    fill!(A, false)
    if fast_scalar_indexing(A)
        @inbounds for i in axes(A, 1)
            A[i, i] = α
        end
    else
        A[diagind(A)] .= α
    end
    return A
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

function __internal_init(
        prob::AbstractNonlinearProblem, alg::TrueJacobianInitialization, solver, f::F, fu,
        u, p; linsolve = missing, internalnorm::IN = DEFAULT_NORM, kwargs...) where {F, IN}
    autodiff = get_concrete_forward_ad(
        alg.autodiff, prob; check_reverse_mode = false, kwargs...)
    jac_cache = JacobianCache(prob, solver, prob.f, fu, u, p; autodiff, linsolve)
    J = alg.structure(jac_cache(nothing))
    return InitializedApproximateJacobianCache(
        J, alg.structure, alg, jac_cache, false, internalnorm)
end

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
@concrete mutable struct InitializedApproximateJacobianCache
    J
    structure
    alg
    cache
    initialized::Bool
    internalnorm
end

function __reinit_internal!(cache::InitializedApproximateJacobianCache, args...; kwargs...)
    cache.initialized = false
end

@internal_caches InitializedApproximateJacobianCache :cache

function (cache::InitializedApproximateJacobianCache)(::Nothing)
    return get_full_jacobian(cache, cache.structure, cache.J)
end

function __internal_solve!(
        cache::InitializedApproximateJacobianCache, fu, u, ::Val{reinit}) where {reinit}
    if reinit || !cache.initialized
        cache(cache.alg, fu, u)
        cache.initialized = true
    end
    if stores_full_jacobian(cache.structure)
        full_J = cache.J
    else
        full_J = get_full_jacobian(cache, cache.structure, cache.J)
    end
    return full_J
end

function (cache::InitializedApproximateJacobianCache)(alg::IdentityInitialization, fu, u)
    α = __initial_alpha(alg.alpha, u, fu, cache.internalnorm)
    cache.J = __make_identity!!(cache.J, α)
    return
end

function (cache::InitializedApproximateJacobianCache)(
        alg::TrueJacobianInitialization, fu, u)
    J_new = cache.cache(u)
    cache.J = cache.structure(cache.J, J_new)
    return
end

# Matrix Inversion
@inline __safe_inv_workspace(A) = nothing, A
@inline __safe_inv_workspace(A::ApplyArray) = __safe_inv_workspace(X)
@inline __safe_inv_workspace(A::SparseMatrixCSC) = Matrix(A), Matrix(A)

@inline __safe_inv!!(workspace, A::Number) = pinv(A)
@inline __safe_inv!!(workspace, A::AbstractMatrix) = pinv(A)
@inline function __safe_inv!!(workspace, A::Diagonal)
    D = A.diag
    @bb @. D = pinv(D)
    return Diagonal(D)
end
@inline function __safe_inv!!(workspace, A::AbstractVector{T}) where {T}
    @. A = ifelse(iszero(A), zero(T), one(T) / A)
    return A
end
@inline __safe_inv!!(workspace, A::ApplyArray) = __safe_inv!!(workspace, A.f(A.args...))
@inline function __safe_inv!!(workspace::AbstractMatrix, A::SparseMatrixCSC)
    copyto!(workspace, A)
    return __safe_inv!!(nothing, workspace)
end
@inline function __safe_inv!!(workspace, A::StridedMatrix{T}) where {T}
    LinearAlgebra.checksquare(A)
    if istriu(A)
        issingular = any(iszero, @view(A[diagind(A)]))
        A_ = UpperTriangular(A)
        !issingular && return triu!(parent(inv(A_)))
    elseif istril(A)
        A_ = LowerTriangular(A)
        issingular = any(iszero, @view(A_[diagind(A_)]))
        !issingular && return tril!(parent(inv(A_)))
    else
        F = lu(A; check = false)
        if issuccess(F)
            Ai = LinearAlgebra.inv!(F)
            return convert(typeof(parent(Ai)), Ai)
        end
    end
    return pinv(A)
end

@inline __safe_inv(x) = __safe_inv!!(first(__safe_inv_workspace(x)), x)

LazyArrays.applied_eltype(::typeof(__safe_inv), x) = eltype(x)
LazyArrays.applied_ndims(::typeof(__safe_inv), x) = ndims(x)
LazyArrays.applied_size(::typeof(__safe_inv), x) = size(x)
LazyArrays.applied_axes(::typeof(__safe_inv), x) = axes(x)
