"""
    DampedNewtonDescent(; linsolve = nothing, precs = DEFAULT_PRECS, initial_damping,
        damping_fn)

A Newton descent algorithm with damping. The damping factor is computed using the
`damping_fn` function. The descent direction is computed as ``(JᵀJ + λDᵀD) δu = -fu``. For
non-square Jacobians, we default to solving for `Jδx = -fu` and `√λ⋅D δx = 0`
simultaneously. If the linear solver can't handle non-square matrices, we use the normal
form equations ``(JᵀJ + λDᵀD) δu = Jᵀ fu``. Note that this factorization is often the faster
choice, but it is not as numerically stable as the least squares solver.

The damping factor returned must be a non-negative number.

### Keyword Arguments

  - `initial_damping`: The initial damping factor to use
  - `damping_fn`: The function to use to compute the damping factor. This must satisfy the
    [`NonlinearSolve.AbstractDampingFunction`](@ref) interface.
"""
@kwdef @concrete struct DampedNewtonDescent <: AbstractDescentAlgorithm
    linsolve = nothing
    precs = DEFAULT_PRECS
    initial_damping
    damping_fn
end

function Base.show(io::IO, d::DampedNewtonDescent)
    modifiers = String[]
    d.linsolve !== nothing && push!(modifiers, "linsolve = $(d.linsolve)")
    d.precs !== DEFAULT_PRECS && push!(modifiers, "precs = $(d.precs)")
    push!(modifiers, "initial_damping = $(d.initial_damping)")
    push!(modifiers, "damping_fn = $(d.damping_fn)")
    print(io, "DampedNewtonDescent($(join(modifiers, ", ")))")
end

supports_line_search(::DampedNewtonDescent) = true
supports_trust_region(::DampedNewtonDescent) = true

@concrete mutable struct DampedNewtonDescentCache{pre_inverted, mode} <:
                         AbstractDescentCache
    J
    δu
    δus
    lincache
    JᵀJ_cache
    Jᵀfu_cache
    rhs_cache
    damping_fn_cache
    timer
end

@internal_caches DampedNewtonDescentCache :lincache :damping_fn_cache

function __internal_init(prob::AbstractNonlinearProblem, alg::DampedNewtonDescent, J, fu,
        u; stats, pre_inverted::Val{INV} = False, linsolve_kwargs = (;),
        abstol = nothing, timer = get_timer_output(), reltol = nothing,
        alias_J = true, shared::Val{N} = Val(1), kwargs...) where {INV, N}
    length(fu) != length(u) &&
        @assert !INV "Precomputed Inverse for Non-Square Jacobian doesn't make sense."
    @bb δu = similar(u)
    δus = N ≤ 1 ? nothing : map(2:N) do i
        @bb δu_ = similar(u)
    end

    normal_form_damping = returns_norm_form_damping(alg.damping_fn)
    normal_form_linsolve = NonlinearSolveBase.needs_square_A(alg.linsolve, u)
    if u isa Number
        mode = :simple
    elseif prob isa NonlinearProblem
        mode = ifelse(!normal_form_damping, :simple,
            ifelse(normal_form_linsolve, :normal_form, :least_squares))
    else
        if normal_form_linsolve & !normal_form_damping
            throw(ArgumentError("Linear Solver expects Normal Form but returned Damping is \
                                 not Normal Form. This is not supported."))
        end
        mode = ifelse(normal_form_damping & !normal_form_linsolve, :least_squares,
            ifelse(!normal_form_damping & !normal_form_linsolve, :simple, :normal_form))
    end

    if mode === :least_squares
        if requires_normal_form_jacobian(alg.damping_fn)
            JᵀJ = transpose(J) * J  # Needed to compute the damping factor
            jac_damp = JᵀJ
        else
            JᵀJ = nothing
            jac_damp = J
        end
        if requires_normal_form_rhs(alg.damping_fn)
            Jᵀfu = transpose(J) * _vec(fu)
            rhs_damp = Jᵀfu
        else
            Jᵀfu = nothing
            rhs_damp = fu
        end
        damping_fn_cache = __internal_init(prob, alg.damping_fn, alg.initial_damping,
            jac_damp, rhs_damp, u, False; stats, kwargs...)
        D = damping_fn_cache(nothing)
        D isa Number && (D = D * I)
        rhs_cache = vcat(_vec(fu), _vec(u))
        J_cache = _vcat(J, D)
        A, b = J_cache, rhs_cache
    elseif mode === :simple
        damping_fn_cache = __internal_init(
            prob, alg.damping_fn, alg.initial_damping, J, fu, u, False; kwargs...)
        J_cache = __maybe_unaliased(J, alias_J)
        D = damping_fn_cache(nothing)
        J_damped = __dampen_jacobian!!(J_cache, J, D)
        J_cache = J_damped
        A, b = J_damped, _vec(fu)
        JᵀJ, Jᵀfu, rhs_cache = nothing, nothing, nothing
    elseif mode === :normal_form
        JᵀJ = transpose(J) * J
        Jᵀfu = transpose(J) * _vec(fu)
        jac_damp = requires_normal_form_jacobian(alg.damping_fn) ? JᵀJ : J
        rhs_damp = requires_normal_form_rhs(alg.damping_fn) ? Jᵀfu : fu
        damping_fn_cache = __internal_init(prob, alg.damping_fn, alg.initial_damping,
            jac_damp, rhs_damp, u, True; stats, kwargs...)
        D = damping_fn_cache(nothing)
        @bb J_cache = similar(JᵀJ)
        @bb @. J_cache = 0
        J_damped = __dampen_jacobian!!(J_cache, JᵀJ, D)
        A, b = __maybe_symmetric(J_damped), _vec(Jᵀfu)
        rhs_cache = nothing
    end

    lincache = construct_linear_solver(
        alg, alg.linsolve, A, b, _vec(u); stats, abstol, reltol, linsolve_kwargs...)

    return DampedNewtonDescentCache{INV, mode}(
        J_cache, δu, δus, lincache, JᵀJ, Jᵀfu, rhs_cache, damping_fn_cache, timer)
end

function __internal_solve!(cache::DampedNewtonDescentCache{INV, mode}, J, fu,
        u, idx::Val{N} = Val(1); skip_solve::Bool = false,
        new_jacobian::Bool = true, kwargs...) where {INV, N, mode}
    δu = get_du(cache, idx)
    skip_solve && return DescentResult(; δu)

    recompute_A = idx === Val(1)

    @static_timeit cache.timer "dampen" begin
        if mode === :least_squares
            if (J !== nothing || new_jacobian) && recompute_A
                INV && (J = inv(J))
                if requires_normal_form_jacobian(cache.damping_fn_cache)
                    @bb cache.JᵀJ_cache = transpose(J) × J
                    jac_damp = cache.JᵀJ_cache
                else
                    jac_damp = J
                end
                if requires_normal_form_rhs(cache.damping_fn_cache)
                    @bb cache.Jᵀfu_cache = transpose(J) × fu
                    rhs_damp = cache.Jᵀfu_cache
                else
                    rhs_damp = fu
                end
                D = __internal_solve!(cache.damping_fn_cache, jac_damp, rhs_damp, False)
                if __can_setindex(cache.J)
                    copyto!(@view(cache.J[1:size(J, 1), :]), J)
                    cache.J[(size(J, 1) + 1):end, :] .= sqrt.(D)
                else
                    cache.J = _vcat(J, sqrt.(D))
                end
            end
            A = cache.J
            if __can_setindex(cache.rhs_cache)
                cache.rhs_cache[1:length(fu)] .= _vec(fu)
                cache.rhs_cache[(length(fu) + 1):end] .= false
            else
                cache.rhs_cache = vcat(_vec(fu), zero(_vec(u)))
            end
            b = cache.rhs_cache
        elseif mode === :simple
            if (J !== nothing || new_jacobian) && recompute_A
                INV && (J = inv(J))
                D = __internal_solve!(cache.damping_fn_cache, J, fu, False)
                cache.J = __dampen_jacobian!!(cache.J, J, D)
            end
            A, b = cache.J, _vec(fu)
        elseif mode === :normal_form
            if (J !== nothing || new_jacobian) && recompute_A
                INV && (J = inv(J))
                @bb cache.JᵀJ_cache = transpose(J) × J
                @bb cache.Jᵀfu_cache = transpose(J) × vec(fu)
                D = __internal_solve!(
                    cache.damping_fn_cache, cache.JᵀJ_cache, cache.Jᵀfu_cache, True)
                cache.J = __dampen_jacobian!!(cache.J, cache.JᵀJ_cache, D)
                A = __maybe_symmetric(cache.J)
            elseif !recompute_A
                @bb cache.Jᵀfu_cache = transpose(J) × vec(fu)
                A = __maybe_symmetric(cache.J)
            else
                A = nothing
            end
            b = _vec(cache.Jᵀfu_cache)
        else
            error("Unknown mode: $(mode)")
        end
    end

    @static_timeit cache.timer "linear solve" begin
        linres = cache.lincache(;
            A, b, reuse_A_if_factorization = !new_jacobian && !recompute_A,
            kwargs..., linu = _vec(δu))
        δu = _restructure(get_du(cache, idx), linres.u)
        if !linres.success
            set_du!(cache, δu, idx)
            return DescentResult(; δu, success = false, linsolve_success = false)
        end
    end

    @bb @. δu *= -1
    set_du!(cache, δu, idx)
    return DescentResult(; δu)
end

# Define special concatenation for certain Array combinations
@inline _vcat(x, y) = vcat(x, y)

# J_cache is allowed to alias J
## Compute ``J + D``
@inline __dampen_jacobian!!(J_cache, J::AbstractSciMLOperator, D) = J + D
@inline __dampen_jacobian!!(J_cache, J::Number, D) = J + D
@inline function __dampen_jacobian!!(J_cache, J::AbstractMatrix, D::AbstractMatrix)
    if __can_setindex(J_cache)
        copyto!(J_cache, J)
        if fast_scalar_indexing(J_cache)
            @simd for i in axes(J_cache, 1)
                @inbounds J_cache[i, i] += D[i, i]
            end
        else
            idxs = diagind(J_cache)
            J_cache[idxs] .= @view(J[idxs]) .+ @view(D[idxs])
        end
        return J_cache
    else
        return @. J + D
    end
end
@inline function __dampen_jacobian!!(J_cache, J::AbstractMatrix, D::Number)
    if __can_setindex(J_cache)
        copyto!(J_cache, J)
        if fast_scalar_indexing(J_cache)
            @simd for i in axes(J_cache, 1)
                @inbounds J_cache[i, i] += D
            end
        else
            idxs = diagind(J_cache)
            J_cache[idxs] .= @view(J[idxs]) .+ D
        end
        return J_cache
    else
        return @. J + D
    end
end
