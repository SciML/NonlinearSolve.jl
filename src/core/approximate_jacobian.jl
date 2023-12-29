# TODO: Trust Region
# TODO: alpha_scaling
@concrete struct ApproximateJacobianSolveAlgorithm{concrete_jac, name} <:
                 AbstractNonlinearSolveAlgorithm{name}
    linesearch
    descent
    update_rule
    reinit_rule
    max_resets::UInt
    initialization
end

@inline concrete_jac(::ApproximateJacobianSolveAlgorithm{CJ}) where {CJ} = CJ

@concrete mutable struct ApproximateJacobianSolveCache{INV, GB, iip} <:
                         AbstractNonlinearSolveCache{iip}
    # Basic Requirements
    fu
    u
    u_cache
    p
    du  # Aliased to `get_du(descent_cache)`
    J   # Aliased to `initialization_cache.J` if !INV
    alg
    prob

    # Internal Caches
    initialization_cache
    descent_cache
    linesearch_cache
    trustregion_cache
    update_rule_cache
    reinit_rule

    inv_workspace

    # Counters
    nf::UInt
    nsteps::UInt
    nresets::UInt
    max_resets::UInt
    maxiters::UInt
    total_time::Float64
    cache_initialization_time::Float64

    # Termination & Tracking
    termination_cache
    trace
    retcode::ReturnCode.T
    force_stop::Bool
end

# Accessors Interface
get_fu(cache::ApproximateJacobianSolveCache) = cache.fu
get_u(cache::ApproximateJacobianSolveCache) = cache.u
set_fu!(cache::ApproximateJacobianSolveCache, fu) = (cache.fu = fu)
set_u!(cache::ApproximateJacobianSolveCache, u) = (cache.u = u)

# NLStats interface
# @inline get_nf(cache::ApproximateJacobianSolveCache) = cache.nf +
#                                                        get_nf(cache.linesearch_cache)
# @inline get_njacs(cache::ApproximateJacobianSolveCache) = get_njacs(cache.initialization_cache)
@inline get_nsteps(cache::ApproximateJacobianSolveCache) = cache.nsteps
# @inline increment_nsteps!(cache::ApproximateJacobianSolveCache) = (cache.nsteps += 1)
# @inline function get_nsolve(cache::ApproximateJacobianSolveCache)
#     cache.linsolve_cache === nothing && return 0
#     return get_nsolve(cache.linsolve_cache)
# end
# @inline function get_nfactors(cache::ApproximateJacobianSolveCache)
#     cache.linsolve_cache === nothing && return 0
#     return get_nfactors(cache.linsolve_cache)
# end

function SciMLBase.__init(prob::AbstractNonlinearProblem{uType, iip},
        alg::ApproximateJacobianSolveAlgorithm, args...; alias_u0 = false,
        maxiters = 1000, abstol = nothing, reltol = nothing, linsolve_kwargs = (;),
        termination_condition = nothing, internalnorm::F = DEFAULT_NORM,
        kwargs...) where {uType, iip, F}
    time_start = time()
    (; f, u0, p) = prob
    u = __maybe_unaliased(u0, alias_u0)
    fu = evaluate_f(prob, u)
    @bb u_cache = copy(u)

    INV = store_inverse_jacobian(alg.update_rule)
    # TODO: alpha = __initial_alpha(alg_.alpha, u, fu, internalnorm)

    linsolve = __getproperty(alg.descent, Val(:linsolve))
    initialization_cache = init(prob, alg.initialization, alg, f, fu, u, p; linsolve)

    abstol, reltol, termination_cache = init_termination_cache(abstol, reltol, u, u,
        termination_condition)
    linsolve_kwargs = merge((; abstol, reltol), linsolve_kwargs)

    J = initialization_cache(nothing)
    inv_workspace, J = INV ? __safe_inv_workspace(J) : (nothing, J)
    descent_cache = init(prob, alg.descent, J, fu, u; abstol, reltol, internalnorm,
        linsolve_kwargs, preinverted = Val(INV))
    du = get_du(descent_cache)

    # if alg.trust_region !== missing && alg.linesearch !== missing
    #     error("TrustRegion and LineSearch methods are algorithmically incompatible.")
    # end

    # if alg.trust_region !== missing
    #     supports_trust_region(alg.descent) || error("Trust Region not supported by \
    #                                                  $(alg.descent).")
    #     trustregion_cache = nothing
    #     linesearch_cache = nothing
    #     GB = :TrustRegion
    #     error("Trust Region not implemented yet!")
    # end

    if alg.linesearch !== missing
        supports_line_search(alg.descent) || error("Line Search not supported by \
                                                    $(alg.descent).")
        linesearch_cache = init(prob, alg.linesearch, f, fu, u, p)
        trustregion_cache = nothing
        GB = :LineSearch
    end

    update_rule_cache = init(prob, alg.update_rule, J, fu, u, du)

    trace = init_nonlinearsolve_trace(alg, u, fu, ApplyArray(__zero, J), du;
        uses_jacobian_inverse = Val(INV), kwargs...)

    cache = ApproximateJacobianSolveCache{INV, GB, iip}(fu, u, u_cache, p, du, J, alg, prob,
        initialization_cache, descent_cache, linesearch_cache, trustregion_cache,
        update_rule_cache, alg.reinit_rule, inv_workspace, UInt(0), UInt(0), UInt(0),
        UInt(alg.max_resets), UInt(maxiters), 0.0, 0.0, termination_cache, trace,
        ReturnCode.Default, false)

    cache.cache_initialization_time = time() - time_start
    return cache
end

function SciMLBase.step!(cache::ApproximateJacobianSolveCache{INV, GB, iip};
        recompute_jacobian::Union{Nothing, Bool} = nothing) where {INV, GB, iip}
    new_jacobian = true
    if get_nsteps(cache) == 0
        # First Step is special ignore kwargs
        J_init = solve!(cache.initialization_cache, cache.u, Val(false))
        cache.J = INV ? __safe_inv!!(cache.inv_workspace, J_init) : J_init
        J = cache.J
    else
        if recompute_jacobian === nothing
            # Standard Step
            reinit = cache.reinit_rule(cache.J)
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
            new_jacobian = false # Jacobian won't be updated in this step
            reinit = false # Override Checks: Unsafe operation
        end

        if reinit
            J_init = solve!(cache.initialization_cache, cache.u, Val(true))
            cache.J = INV ? __safe_inv!!(cache.inv_workspace, J_init) : J_init
            J = cache.J
        else
            J = cache.J
        end
    end

    if GB === :LineSearch
        δu = solve!(cache.descent_cache, ifelse(new_jacobian, J, nothing), cache.fu)
        needs_reset, α = solve!(cache.linesearch_cache, cache.u, δu)  # TODO: use `needs_reset`
        @bb axpy!(α, δu, cache.u)
    elseif GB === :TrustRegion
        error("Trust Region not implemented yet!")
    else
        error("Unknown Globalization Strategy: $(GB). Allowed values are (:LineSearch, \
               :TrustRegion)")
    end

    evaluate_f!(cache, cache.u, cache.p)

    # TODO: update_trace!(cache, α)
    check_and_update!(cache, cache.fu, cache.u, cache.u_cache)

    @bb copyto!(cache.u_cache, cache.u)

    (cache.force_stop || (recompute_jacobian !== nothing && !recompute_jacobian)) &&
        return nothing

    cache.J = solve!(cache.update_rule_cache, cache.J, cache.fu, cache.u, δu)

    return nothing
end

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
    return InitializedApproximateJacobianCache(one(u), alg.structure, alg, nothing, true,
        0.0)
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
    return InitializedApproximateJacobianCache(J, alg.structure, alg, nothing, true, 0.0)
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

# @inline function get_njacs(cache::InitializedApproximateJacobianCache)
#     cache.cache === nothing && return 0
#     return get_njacs(cache.cache)
# end

function (cache::InitializedApproximateJacobianCache)(::Nothing)
    return get_full_jacobian(cache, cache.structure, cache.J)
end

function SciMLBase.solve!(cache::InitializedApproximateJacobianCache, u,
        ::Val{reinit}) where {reinit}
    time_start = time()
    if reinit || !cache.initialized
        cache(cache.alg, u)
        cache.initialized = true
    end
    if stores_full_jacobian(cache.structure)
        full_J = cache.J
    else
        full_J = get_full_jacobian(cache, cache.structure, cache.J)
    end
    cache.total_time += time() - time_start
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
