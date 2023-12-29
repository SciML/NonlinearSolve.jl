"""
    DampedNewtonDescent(; linsolve = nothing, precs = DEFAULT_PRECS, initial_damping,
        damping_fn)

A Newton descent algorithm with damping. The damping factor is computed using the
`damping_fn` function. The descent direction is computed as ``(JᵀJ + λDᵀD) δu = -fu``. For
non-square Jacobians, we default to solving for `Jδx = -fu` and `√λ⋅D δx = 0`
simultaneously. If the linear solver can't handle non-square matrices, we use the normal
form equations ``(JᵀJ + λDᵀD) δu = Jᵀ fu``. Note that this factorization is often the faster
choice, but it is not as numerically stable as the least squares solver.

Based on the formulation we expect the damping factor returned to be a non-negative number.
"""
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
        reltol = nothing, alias_J = true, shared::Val{N} = Val(1), kwargs...) where {N, INV}
    length(fu) != length(u) &&
        @assert !INV "Precomputed Inverse for Non-Square Jacobian doesn't make sense."
    @bb δu = similar(u)
    δus = N ≤ 1 ? nothing : map(2:N) do i
        @bb δu_ = similar(u)
    end
    normal_form = __needs_square_A(alg.linsolve, u)

    if normal_form
        JᵀJ = transpose(J) * J
        Jᵀfu = transpose(J) * _vec(fu)
        damping_fn_cache = init(prob, alg.damping_fn, alg.initial_damping, JᵀJ, fu, u;
            kwargs...)
        D = solve!(damping_fn_cache, JᵀJ, Jᵀfu)
        @bb J_cache = similar(JᵀJ)
        J_damped = __dampen_jacobian!!(J_cache, JᵀJ, D)
        A, b = __maybe_symmetric(J_damped), _vec(Jᵀfu)
    else
        JᵀJ = transpose(J) * J  # Needed to compute the damping factor
        damping_fn_cache = init(prob, alg.damping_fn, alg.initial_damping, JᵀJ, fu, u;
            kwargs...)
        D = solve!(damping_fn_cache, JᵀJ, fu)
        D isa Number && (D = D * I)
        Jᵀfu = vcat(_vec(fu), _vec(u))
        J_cache = _vcat(J, D)
        A, b = J_cache, Jᵀfu
    end

    lincache = LinearSolverCache(alg, alg.linsolve, A, b, _vec(u); abstol, reltol,
        linsolve_kwargs...)

    return DampedNewtonDescentCache{INV, true, normal_form}(J_cache, δu, δus, lincache, JᵀJ,
        Jᵀfu, damping_fn_cache)
end

# Define special concatenation for certain Array combinations
@inline _vcat(x, y) = vcat(x, y)

function SciMLBase.solve!(cache::DampedNewtonDescentCache{INV, false}, J, fu, u,
        idx::Val{N} = Val(1); skip_solve::Bool = false, kwargs...) where {INV, N}
    δu = get_du(cache, idx)
    skip_solve && return δu
    if J !== nothing
        INV && (J = inv(J))
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
    δu = get_du(cache, idx)
    skip_solve && return δu

    if normal_form
        if J !== nothing
            INV && (J = inv(J))
            @bb cache.JᵀJ_cache = transpose(J) × J
            D = solve!(cache.damping_fn_cache, cache.JᵀJ_cache, fu)
            @bb cache.Jᵀfu_cache = transpose(J) × vec(fu)
            J_ = __dampen_jacobian!!(cache.J, cache.JᵀJ_cache, D)
        else
            J_ = cache.JᵀJ_cache
        end
        δu = cache.lincache(; A = __maybe_symmetric(J_), b = cache.Jᵀfu_cache, kwargs...,
            linu = _vec(δu))
    else
        if J !== nothing
            INV && (J = inv(J))
            @bb cache.JᵀJ_cache = transpose(J) × J
            # FIXME: We can compute the damping factor without the Jacobian
            D = solve!(cache.damping_fn_cache, cache.JᵀJ_cache, fu)
            if can_setindex(cache.J)
                copyto!(@view(cache.J[1:size(J, 1), :]), J)
                cache.J[(size(J, 1) + 1):end, :] .= sqrt.(D)
            else
                cache.J = _vcat(J, sqrt.(D))
            end
            if can_setindex(cache.Jᵀfu_cache)
                cache.Jᵀfu_cache[1:size(J, 1)] .= _vec(fu)
                cache.Jᵀfu_cache[(size(J, 1) + 1):end] .= false
            else
                cache.Jᵀfu_cache = vcat(_vec(fu), zero(_vec(u)))
            end
        end
        A, b = cache.J, cache.Jᵀfu_cache
        δu = cache.lincache(; A, b, kwargs..., linu = _vec(δu))
    end

    δu = _restructure(get_du(cache, idx), δu)
    @bb @. δu *= -1
    set_du!(cache, δu, idx)
    return δu, true, (;)
end

# J_cache is allowed to alias J
## Compute ``J - D``
@inline __dampen_jacobian!!(J_cache, J::SciMLBase.AbstractSciMLOperator, D) = J + D
@inline __dampen_jacobian!!(J_cache, J::Number, D) = J + D
@inline function __dampen_jacobian!!(J_cache, J::AbstractArray, D)
    if can_setindex(J_cache)
        D_ = diag(D)
        if fast_scalar_indexing(J_cache)
            @inbounds for i in axes(J_cache, 1)
                J_cache[i, i] = J[i, i] + D_[i]
            end
        else
            idxs = diagind(J_cache)
            @.. broadcast=false @view(J_cache[idxs])=@view(J[idxs]) + D_
        end
        return J_cache
    else
        return @. J - D
    end
end
@inline function __dampen_jacobian!!(J_cache, J::AbstractArray, D::Number)
    if can_setindex(J_cache)
        if fast_scalar_indexing(J_cache)
            @inbounds for i in axes(J_cache, 1)
                J_cache[i, i] = J[i, i] + D
            end
        else
            idxs = diagind(J_cache)
            @.. broadcast=false @view(J_cache[idxs])=@view(J[idxs]) + D
        end
        return J_cache
    else
        return @. J - D
    end
end
