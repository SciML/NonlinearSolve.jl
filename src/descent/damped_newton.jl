@kwdef @concrete struct DampedNewtonDescent <: AbstractDescentAlgorithm
    linsolve = nothing
    precs = DEFAULT_PRECS
    initial_damping
    damping_fn
end

supports_line_search(::DampedNewtonDescent) = true

@concrete mutable struct DampedNewtonDescentCache{pre_inverted, ls, normalform} <:
                         AbstractDescentCache
    J
    δu
    δus
    lincache
    JᵀJ_cache
    Jᵀfu_cache
    damping_fn_cache
end

function SciMLBase.init(prob::NonlinearProblem, alg::DampedNewtonDescent, J, fu, u;
        pre_inverted::Val{INV} = False, linsolve_kwargs = (;), abstol = nothing,
        reltol = nothing, alias_J = true, shared::Val{N} = Val(1), kwargs...) where {INV, N}
    damping_fn_cache = init(prob, alg.damping_fn, alg.initial_damping, J, fu, u;
        kwargs...)
    @bb δu = similar(u)
    δus = N ≤ 1 ? nothing : map(2:N) do i
        @bb δu_ = similar(u)
    end
    J_cache = __maybe_unaliased(J, alias_J)
    D = solve!(damping_fn_cache, J, fu)
    J_damped = __dampen_jacobian!!(J_cache, J, D)
    lincache = LinearSolverCache(alg, alg.linsolve, J_damped, _vec(fu), _vec(u); abstol,
        reltol, linsolve_kwargs...)
    return DampedNewtonDescentCache{INV, false, false}(J, δu, δus, lincache, nothing,
        nothing, damping_fn_cache)
end

function SciMLBase.init(prob::NonlinearLeastSquaresProblem, alg::DampedNewtonDescent, J, fu,
        u; pre_inverted::Val{INV} = False, linsolve_kwargs = (;), abstol = nothing,
        reltol = nothing, alias_J = true, kwargs...) where {INV}
    error("Not Implemented Yet!")
end

function SciMLBase.solve!(cache::DampedNewtonDescentCache{INV, false}, J, fu, u,
        idx::Val{N} = Val(1); skip_solve::Bool = false, kwargs...) where {INV, N}
    δu = get_du(cache, idx)
    skip_solve && return δu
    if INV
        @assert J!==nothing "`J` must be provided when `pre_inverted = Val(true)`."
        J = inv(J)
    end
    if J !== nothing
        D = solve!(cache.damping_fn_cache, J, fu)
        J_ = __dampen_jacobian!!(cache.J, J, D)
    else # Use the old factorization
        J_ = J
    end
    δu = cache.lincache(; A = J_, b = _vec(fu), kwargs..., linu = _vec(δu))
    δu = _restructure(get_du(cache, idx), δu)
    @bb @. δu *= -1
    set_du!(cache, δu, idx)
    return δu, true, (;)
end

function SciMLBase.solve!(cache::DampedNewtonDescentCache{INV, true, normal_form}, J, fu, u,
        idx::Val{N} = Val(1); skip_solve::Bool = false,
        kwargs...) where {INV, normal_form, N}
    skip_solve && return cache.δu
    error("Not Implemented Yet!")
end

# J_cache is allowed to alias J
## Compute ``J - D``
@inline __dampen_jacobian!!(J_cache, J::SciMLBase.AbstractSciMLOperator, D) = J - D
@inline  __dampen_jacobian!!(J_cache, J::Number, D) = J - D
@inline function __dampen_jacobian!!(J_cache, J::AbstractArray, D)
    if can_setindex(J_cache)
        D_ = diag(D)
        if fast_scalar_indexing(J_cache)
            @inbounds for i in axes(J_cache, 1)
                J_cache[i, i] = J[i, i] - D_[i]
            end
        else
            idxs = diagind(J_cache)
            @.. broadcast=false @view(J_cache[idxs])=@view(J[idxs]) - D_
        end
        return J_cache
    else
        return @. J - D
    end
end
@inline function __dampen_jacobian!!(J_cache, J::AbstractArray, D::UniformScaling)
    if can_setindex(J_cache)
        if fast_scalar_indexing(J_cache)
            @inbounds for i in axes(J_cache, 1)
                J_cache[i, i] = J[i, i] - D.λ
            end
        else
            idxs = diagind(J_cache)
            @.. broadcast=false @view(J_cache[idxs])=@view(J[idxs]) - D.λ
        end
        return J_cache
    else
        return @. J - D
    end
end
