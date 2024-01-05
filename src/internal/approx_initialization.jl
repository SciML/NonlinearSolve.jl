# Jacobian Structure
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
    if can_setindex(J)
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

struct FullStructure <: AbstractApproximateJacobianStructure end

stores_full_jacobian(::FullStructure) = true

(::FullStructure)(J; alias::Bool = false) = alias ? J : @bb(copy(J))

function (::FullStructure)(J, J_new)
    J === J_new && return J
    @bb copyto!(J, J_new)
    return J
end

# Initialization Strategies
@concrete struct IdentityInitialization <: AbstractJacobianInitialization
    structure
end

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::IdentityInitialization, solver,
        f::F, fu, u::Number, p; kwargs...) where {F}
    return InitializedApproximateJacobianCache(one(u), alg.structure, alg, nothing, true)
end
function SciMLBase.init(prob::AbstractNonlinearProblem, alg::IdentityInitialization, solver,
        f::F, fu::StaticArray, u::StaticArray, p; kwargs...) where {F}
    if alg.structure isa DiagonalStructure
        @assert length(u)==length(fu) "Diagonal Jacobian Structure must be square!"
        J = one.(fu)
    else
        T = promote_type(eltype(u), eltype(fu))
        if fu isa SArray
            J_ = SArray{Tuple{prod(Size(fu)), prod(Size(u))}, T}(I)
        else
            J_ = MArray{Tuple{prod(Size(fu)), prod(Size(u))}, T}(I)
        end
        J = alg.structure(J_; alias = true)
    end
    return InitializedApproximateJacobianCache(J, alg.structure, alg, nothing, true)
end
function SciMLBase.init(prob::AbstractNonlinearProblem, alg::IdentityInitialization, solver,
        f::F, fu, u, p; kwargs...) where {F}
    if alg.structure isa DiagonalStructure
        @assert length(u)==length(fu) "Diagonal Jacobian Structure must be square!"
        J = one.(fu)
    else
        J_ = similar(fu, promote_type(eltype(fu), eltype(u)), length(fu), length(u))
        J = alg.structure(__make_identity!!(J_); alias = true)
    end
    return InitializedApproximateJacobianCache(J, alg.structure, alg, nothing, true)
end

@inline __make_identity!!(A::Number) = one(A)
@inline __make_identity!!(A::AbstractVector) = can_setindex(A) ? (A .= true) : (one.(A))
@inline function __make_identity!!(A::AbstractMatrix{T}) where {T}
    if A isa SMatrix
        Sz = Size(A)
        return SArray{Tuple{Sz[1], Sz[2]}, eltype(Sz)}(I)
    end
    @assert can_setindex(A) "__make_identity!!(::AbstractMatrix) only works on mutable arrays!"
    fill!(A, false)
    if fast_scalar_indexing(A)
        @inbounds for i in axes(A, 1)
            A[i, i] = true
        end
    else
        A[diagind(A)] .= true
    end
    return A
end

@concrete struct TrueJacobianInitialization <: AbstractJacobianInitialization
    structure
    autodiff
end

# TODO: For just the diagonal elements of the Jacobian we don't need to construct the full
# Jacobian
function SciMLBase.init(prob::AbstractNonlinearProblem, alg::TrueJacobianInitialization,
        solver, f::F, fu, u, p; linsolve = missing, autodiff = nothing, kwargs...) where {F}
    autodiff = get_concrete_forward_ad(alg.autodiff, prob; check_reverse_mode = false,
        kwargs...)
    jac_cache = JacobianCache(prob, solver, prob.f, fu, u, p; autodiff, linsolve)
    J = alg.structure(jac_cache(nothing))
    return InitializedApproximateJacobianCache(J, alg.structure, alg, jac_cache, false)
end

@concrete mutable struct InitializedApproximateJacobianCache
    J
    structure
    alg
    cache
    initialized::Bool
end

@internal_caches InitializedApproximateJacobianCache :cache

function (cache::InitializedApproximateJacobianCache)(::Nothing)
    return get_full_jacobian(cache, cache.structure, cache.J)
end

function SciMLBase.solve!(cache::InitializedApproximateJacobianCache, u,
        ::Val{reinit}) where {reinit}
    if reinit || !cache.initialized
        cache(cache.alg, u)
        cache.initialized = true
    end
    if stores_full_jacobian(cache.structure)
        full_J = cache.J
    else
        full_J = get_full_jacobian(cache, cache.structure, cache.J)
    end
    return full_J
end

function (cache::InitializedApproximateJacobianCache)(alg::IdentityInitialization, u)
    cache.J = __make_identity!!(cache.J)
    return
end

function (cache::InitializedApproximateJacobianCache)(alg::TrueJacobianInitialization, u)
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
        A_ = UpperTriangular(A)
        issingular = any(iszero, @view(A_[diagind(A_)]))
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
