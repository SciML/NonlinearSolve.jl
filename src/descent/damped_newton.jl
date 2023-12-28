@kwdef @concrete struct DampedNewtonDescent <: AbstractDescentAlgorithm
    linsolve = nothing
    precs = DEFAULT_PRECS
    initial_damping
    damping_fn
end

supports_line_search(::DampedNewtonDescent) = true
supports_trust_region(::DampedNewtonDescent) = true

@concrete mutable struct DampedNewtonDescent{pre_inverted, ls, normalform} <:
                         AbstractDescentCache
    J
    δu
    lincache
    JᵀJ_cache
    Jᵀfu_cache
end

function SciMLBase.init(prob::NonlinearProblem, alg::DampedNewtonDescent, J, fu, u;
        pre_inverted::Val{INV} = False, linsolve_kwargs = (;), abstol = nothing,
        reltol = nothing, alias_J = true, kwargs...) where {INV}
    damping_fn_cache = init(prob, alg.damping_update_fn, alg.initial_damping, J, fu, u;
        kwargs...)
    @bb δu = similar(u)
    J_cache = __maybe_unaliased(J, alias_J)
    J_damped = __dampen_jacobian!!(J_cache, J, damping_fn_cache)
    lincache = LinearSolverCache(alg, alg.linsolve, J_damped, _vec(fu), _vec(u); abstol,
        reltol, linsolve_kwargs...)
    return DampedNewtonDescent{INV, false, false}(J, δu, lincache, nothing, nothing)
end

function SciMLBase.init(prob::NonlinearLeastSquaresProblem, alg::DampedNewtonDescent, J, fu,
        u; pre_inverted::Val{INV} = False, linsolve_kwargs = (;), abstol = nothing,
        reltol = nothing, alias_J = true, kwargs...) where {INV}
    error("Not Implemented Yet!")
#     @assert !INV "Precomputed Inverse for Non-Square Jacobian doesn't make sense."

#     normal_form = __needs_square_A(alg.linsolve, u)
#     if normal_form
#         JᵀJ = transpose(J) * J
#         Jᵀfu = transpose(J) * _vec(fu)
#         A, b = __maybe_symmetric(JᵀJ), Jᵀfu
#     else
#         JᵀJ, Jᵀfu = nothing, nothing
#         A, b = J, _vec(fu)
#     end
#     lincache = LinearSolveCache(alg, alg.linsolve, A, b, _vec(u); abstol, reltol,
#         linsolve_kwargs...)
#     @bb δu = similar(u)
#     return NewtonDescentCache{false, normal_form}(δu, lincache, JᵀJ, Jᵀfu)
end

function SciMLBase.solve!(cache::DampedNewtonDescentCache{INV, false, false}, J, fu;
        skip_solve::Bool = false, kwargs...) where {INV}
    skip_solve && return cache.δu
    if INV
        @assert J!==nothing "`J` must be provided when `pre_inverted = Val(true)`."
        J = inv(J)
    end
    D = solve!(cache.damping_fn_cache, J, fu)
    J_ = __dampen_jacobian!!(cache.J, J, D)
    δu = cache.lincache(; A = J_, b = _vec(fu), kwargs..., du = _vec(cache.δu))
    cache.δu = _restructure(cache.δu, δu)
    @bb @. cache.δu *= -1
    return cache.δu
end

# function SciMLBase.solve!(cache::NewtonDescentCache{false, true}, J, fu;
#         skip_solve::Bool = false, kwargs...)
#     skip_solve && return cache.δu
#     @bb cache.JᵀJ_cache = transpose(J) × J
#     @bb cache.Jᵀfu_cache = transpose(J) × fu
#     δu = cache.lincache(; A = cache.JᵀJ_cache, b = cache.Jᵀfu_cache, kwargs...,
#         du = _vec(cache.δu))
#     cache.δu = _restructure(cache.δu, δu)
#     @bb @. cache.δu *= -1
#     return cache.δu
# end

# J_cache is allowed to alias J
## Compute ``J - D``
@inline __dampen_jacobian!!(J_cache, J::SciMLBase.AbstractSciMLOperator, D) = J - D
@inline function __dampen_jacobian!!(J_cache, J, D)
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
