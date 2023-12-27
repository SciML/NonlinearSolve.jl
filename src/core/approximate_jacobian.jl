abstract type AbstractApproximateJacobianSolveAlgorithm <: AbstractNonlinearSolveAlgorithm end

abstract type AbstractApproximateJacobianUpdateRule{INV} end

__store_inverse_jacobian(::AbstractApproximateJacobianUpdateRule{INV}) where {INV} = INV

abstract type AbstractApproximateJacobianStructure end
abstract type AbstractJacobianInitialization end

# TODO: alpha_scaling
@concrete struct ApproximateJacobianSolveAlgorithm{INV, I <: AbstractJacobianInitialization,
    D <: AbstractJacobianDampingStrategy, LS <: AbstractNonlinearSolveLineSearchAlgorithm,
    UR <: AbstractApproximateJacobianUpdateRule} <: AbstractApproximateJacobianSolveAlgorithm
    name::Symbol
    initialization::I
    damping::D
    linesearch::LS
    update_rule::UR
    reinit_rule
    linsolve
    max_resets::UInt
end

@inline __concrete_jac(::ApproximateJacobianSolveAlgorithm) = true

@concrete mutable struct ApproximateJacobianSolveCache{INV, iip} <:
                         AbstractNonlinearSolveCache{iip}
    # Basic Requirements
    fu
    u
    u_cache
    p
    du
    alg
    prob

    # Internal Caches
    initialization_cache
    damping_cache
    linesearch_cache
    update_rule_cache
    linsolve_cache
    reinit_rule_cache

    # Algorithm Specific Cache
    inv_workspace
    J  ## Could be J or J⁻¹ based on INV

    # Counters
    nf::UInt
    nsteps::UInt
    nresets::UInt
    max_resets::UInt
    total_time::Float64
    cache_initialization_time::Float64

    # Termination & Tracking
    termination_cache
    tracing_cache
    retcode::ReturnCode.T
    force_stop::Bool
end

# NLStats interface
@inline get_nf(cache::ApproximateJacobianSolveCache) = cache.nf
@inline get_njacs(cache::ApproximateJacobianSolveCache) = get_njacs(cache.initialization_cache)
@inline get_nsteps(cache::ApproximateJacobianSolveCache) = cache.nsteps
@inline increment_nsteps!(cache::ApproximateJacobianSolveCache) = (cache.nsteps += 1)
@inline function get_nsolve(cache::ApproximateJacobianSolveCache)
    cache.linsolve_cache === nothing && return 0
    return get_nsolve(cache.linsolve_cache)
end
@inline function get_nfactors(cache::ApproximateJacobianSolveCache)
    cache.linsolve_cache === nothing && return 0
    return get_nfactors(cache.linsolve_cache)
end

function SciMLBase.__init(prob::AbstractNonlinearProblem{uType, iip},
        alg::ApproximateJacobianSolveAlgorithm{INV}, args...; alias_u0 = false,
        maxiters = 1000, abstol = nothing, reltol = nothing, linsolve_kwargs = (;),
        termination_condition = nothing, internalnorm::F = DEFAULT_NORM,
        kwargs...) where {uType, iip, F, INV}
    time_start = time()
    (; f, u0, p) = prob
    u = __maybe_unaliased(u0, alias_u0)
    # TODO: fu = evaluate_f(prob, u)
    du = @bb similar(u)
    u_cache = @bb copy(u)

    # TODO: alpha = __initial_alpha(alg_.alpha, u, fu, internalnorm)

    initialization_cache = alg.initialization(prob, alg, f, fu, u, p)
    # NOTE: Damping is use for the linear solve if needed but the updates are not performed
    #       on the damped Jacobian
    damping_cache = alg.damping(initialization_cache.J; alias = false)
    inv_workspace, J = INV ? __safe_inv_workspace(damping_cache.J) :
                       (nothing, damping_cache.J)
    # TODO: linesearch_cache
    # TODO: update_rule_cache
    # TODO: linsolve_cache
    # TODO: reinit_rule_cache

    # TODO: termination_cache
    # TODO: tracing_cache

    return ApproximateJacobianSolveCache{iip}(fu, u, u_cache, p, du, alg, prob,
        initialization_cache, damping_cache, linesearch_cache, update_rule_cache,
        linsolve_cache, reinit_rule_cache, inv_workspace, J, 0, 0, 0, alg.max_resets, 0.0,
        time() - time_start, termination_cache, tracing_cache, ReturnCode.Default, false)
end

function SciMLBase.step!(cache::ApproximateJacobianSolveCache{INV, iip};
        recompute_jacobian::Union{Nothing, Bool} = nothing) where {INV, iip}
    if get_nsteps(cache) == 0
        # First Step is special ignore kwargs
        J_init = cache.initialization_cache(cache.initialization, Val(false))
        J_damp = cache.damping_cache(J_init)
        cache.J = INV ? __safe_inv!!(cache.inv_workspace, J_damp) : J_damp
        J = cache.J
    else
        if recompute_jacobian === nothing
            # Standard Step
            reinit = cache.reinit_rule_cache(cache.J)
            if reinit
                cache.nresets += 1
                if cache.nresets ≥ cache.max_resets
                    cache.retcode = ReturnCode.ConvergenceFailure
                    cache.force_stop = true
                    return
                end
            end
        elseif recompute_jacobian
            reinit = true  # Force ReInitialization: Don't count towards resetting
        else
            reinit = false # Override Checks: Unsafe operation
        end
        
        if reinit
            J_ = cache.initialization_cache(cache.initialization, Val(true))
            J_damp = cache.damping_cache(J_)
            cache.J = INV ? __safe_inv!!(cache.inv_workspace, J_damp) : J_damp
            J = cache.J
        else
            J = cache.damping_cache(cache.J, Val(INV))
        end
    end

    # TODO: Perform Linear Solve or Matrix Multiply
    # TODO: Perform Line Search
    # TODO: Update `u` and `fu`

    # TODO: Tracing
    # TODO: Termination
    # TODO: Copy

    cache.force_stop && return nothing

    # TODO: Update the Jacobian

    return nothing
end

# Jacobian Structure
struct DiagonalStructure <: AbstractApproximateJacobianStructure end

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
    end
    return diag(J_new)
end
function (st::DiagonalStructure)(J::AbstractArray, J_new::AbstractMatrix)
    return _restructure(J, st(vec(J), J_new))
end

struct FullStructure <: AbstractApproximateJacobianStructure end

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

function (alg::IdentityInitialization)(prob, alg, f::F, fu, u::Number, p) where {F}
    return InitializedApproximateJacobianCache(one(u), alg.structure, alg, nothing, true,
        0.0)
end
function (alg::IdentityInitialization)(prob, alg, f::F, fu::StaticArray,
        u::StaticArray, p) where {F}
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
    return InitializedApproximateJacobianCache(J, alg.structure, alg, nothing, true, 0.0)
end
function (alg::IdentityInitialization)(prob, f::F, fu, alg, u, p) where {F}
    if alg.structure isa DiagonalStructure
        @assert length(u)==length(fu) "Diagonal Jacobian Structure must be square!"
        J = one.(fu)
    else
        J_ = similar(fu, promote_type(eltype(fu), eltype(u)), length(fu), length(u))
        J = alg.structure(__make_identity!!(J_); alias = true)
    end
    return InitializedApproximateJacobianCache(J, alg.structure, alg, nothing, true, 0.0)
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
end

# TODO: For just the diagonal elements of the Jacobian we don't need to construct the full
# Jacobian
function (alg::TrueJacobianInitialization)(prob, alg, f::F, fu, u, p) where {F}
    jac_cache = JacobianCache(prob, alg, prob.f, fu, u, p)
    J = alg.structure(jac_cache.J)
    return InitializedApproximateJacobianCache(J, alg.structure, alg, jac_cache, false, 0.0)
end

@concrete mutable struct InitializedApproximateJacobianCache
    J
    structure
    alg
    cache
    initialized::Bool
    total_time::Float64
end

@inline function get_njacs(cache::InitializedApproximateJacobianCache)
    cache.cache === nothing && return 0
    return get_njacs(cache.cache)
end

function (cache::InitializedApproximateJacobianCache)(u, ::Val{reinit}) where {reinit}
    time_start = time()
    if reinit || !cache.initialized
        cache(cache.alg, u)
        cache.initialized = true
    end
    cache.total_time += time() - time_start
    return cache.J
end

function (cache::InitializedApproximateJacobianCache)(alg::IdentityInitialization, u)
    cache.J = __make_identity!!(cache.J)
    return
end

function (cache::InitializedApproximateJacobianCache)(alg::TrueJacobianInitialization, u)
    J_new = cache.cache(u)
    cache.J = cache.structure(J_new, cache.J)
    return
end

# Matrix Inversion
@inline __safe_inv_workspace(A) = nothing, A
@inline __safe_inv_workspace(A::ApplyArray) = __safe_inv_workspace(X)
@inline __safe_inv_workspace(A::SparseMatrixCSC) = Matrix(A), Matrix(A)

@inline __safe_inv!!(workspace, A::Number) = pinv(A)
@inline __safe_inv!!(workspace, A::AbstractMatrix) = pinv(A)
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
