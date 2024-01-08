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
    timer::TimerOutput
end

@internal_caches DampedNewtonDescentCache :lincache :damping_fn_cache

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::DampedNewtonDescent, J, fu, u;
        pre_inverted::Val{INV} = False, linsolve_kwargs = (;), abstol = nothing,
        timer = TimerOutput(), reltol = nothing, alias_J = true, shared::Val{N} = Val(1),
        kwargs...) where {INV, N}
    length(fu) != length(u) &&
        @assert !INV "Precomputed Inverse for Non-Square Jacobian doesn't make sense."
    @bb δu = similar(u)
    δus = N ≤ 1 ? nothing : map(2:N) do i
        @bb δu_ = similar(u)
    end

    normal_form_linsolve = __needs_square_A(alg.linsolve, u)
    normal_form_damping = returns_norm_form_damping(alg.damping_fn)

    if normal_form_linsolve & !normal_form_damping
        throw(ArgumentError("Linear Solver expects Normal Form but returned Damping is not \
                             Normal Form. This is not supported."))
    end

    mode = ifelse(normal_form_damping & !normal_form_linsolve, :least_squares,
        ifelse(!normal_form_damping & !normal_form_linsolve, :simple, :normal_form))

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
        damping_fn_cache = init(prob, alg.damping_fn, alg.initial_damping, jac_damp,
            rhs_damp, u, False; kwargs...)
        D = damping_fn_cache(nothing)
        D isa Number && (D = D * I)
        rhs_cache = vcat(_vec(fu), _vec(u))
        J_cache = _vcat(J, D)
        A, b = J_cache, rhs_cache
    elseif mode === :simple
        damping_fn_cache = init(prob, alg.damping_fn, alg.initial_damping, J, fu, u, False;
            kwargs...)
        J_cache = __maybe_unaliased(J, alias_J)
        D = damping_fn_cache(nothing)
        J_damped = __dampen_jacobian!!(J_cache, J, D)
        A, b = J_damped, _vec(fu)
        JᵀJ, Jᵀfu, rhs_cache = nothing, nothing, nothing
    elseif mode === :normal_form
        JᵀJ = transpose(J) * J
        Jᵀfu = transpose(J) * _vec(fu)
        jac_damp = requires_normal_form_jacobian(alg.damping_fn) ? JᵀJ : J
        rhs_damp = requires_normal_form_rhs(alg.damping_fn) ? Jᵀfu : fu
        damping_fn_cache = init(prob, alg.damping_fn, alg.initial_damping, jac_damp,
            rhs_damp, u, True; kwargs...)
        D = damping_fn_cache(nothing)
        @bb J_cache = similar(JᵀJ)
        @bb @. J_cache = 0
        J_damped = __dampen_jacobian!!(J_cache, JᵀJ, D)
        A, b = __maybe_symmetric(J_damped), _vec(Jᵀfu)
        rhs_cache = nothing
    end

    lincache = LinearSolverCache(alg, alg.linsolve, A, b, _vec(u); abstol, reltol,
        linsolve_kwargs...)

    return DampedNewtonDescentCache{INV, mode}(J_cache, δu, δus, lincache, JᵀJ, Jᵀfu,
        rhs_cache, damping_fn_cache, timer)
end

function SciMLBase.solve!(cache::DampedNewtonDescentCache{INV, mode}, J, fu, u,
        idx::Val{N} = Val(1); skip_solve::Bool = false, kwargs...) where {INV, N, mode}
    δu = get_du(cache, idx)
    skip_solve && return δu, true, (;)

    recompute_A = idx === Val(1)

    @timeit_debug cache.timer "dampen" begin
        if mode === :least_squares
            if J !== nothing && recompute_A
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
                D = solve!(cache.damping_fn_cache, jac_damp, rhs_damp, False)
                if __can_setindex(cache.J)
                    copyto!(@view(cache.J[1:size(J, 1), :]), J)
                    cache.J[(size(J, 1) + 1):end, :] .= sqrt.(D)
                else
                    cache.J = _vcat(J, sqrt.(D))
                end
                A = cache.J
            else
                A = nothing
            end
            if __can_setindex(cache.Jᵀfu_cache)
                cache.rhs_cache[1:length(fu)] .= _vec(fu)
                cache.rhs_cache[(length(fu) + 1):end] .= false
            else
                cache.rhs_cache = vcat(_vec(fu), zero(_vec(u)))
            end
            b = cache.rhs_cache
        elseif mode === :simple
            if J !== nothing && recompute_A
                INV && (J = inv(J))
                D = solve!(cache.damping_fn_cache, J, fu, False)
                J_ = __dampen_jacobian!!(cache.J, J, D)
            else # Use the old factorization
                J_ = nothing
            end
            A, b = J_, _vec(fu)
        elseif mode === :normal_form
            if J !== nothing && recompute_A
                INV && (J = inv(J))
                @bb cache.JᵀJ_cache = transpose(J) × J
                @bb cache.Jᵀfu_cache = transpose(J) × vec(fu)
                D = solve!(cache.damping_fn_cache, cache.JᵀJ_cache, cache.Jᵀfu_cache, True)
                J_ = __dampen_jacobian!!(cache.J, cache.JᵀJ_cache, D)
                A = __maybe_symmetric(J_)
            elseif !recompute_A
                @bb cache.Jᵀfu_cache = transpose(J) × vec(fu)
                A = nothing
            else
                A = nothing
            end
            b = _vec(cache.Jᵀfu_cache)
        else
            error("Unknown mode: $(mode)")
        end
    end

    @timeit_debug cache.timer "linear solve" begin
        δu = cache.lincache(; A, b, kwargs..., linu = _vec(δu))
        δu = _restructure(get_du(cache, idx), δu)
    end

    @bb @. δu *= -1
    set_du!(cache, δu, idx)
    return δu, true, (;)
end

# Define special concatenation for certain Array combinations
@inline _vcat(x, y) = vcat(x, y)

# J_cache is allowed to alias J
## Compute ``J + D``
@inline __dampen_jacobian!!(J_cache, J::SciMLBase.AbstractSciMLOperator, D) = J + D
@inline __dampen_jacobian!!(J_cache, J::Number, D) = J + D
@inline function __dampen_jacobian!!(J_cache, J::AbstractMatrix, D::AbstractMatrix)
    if __can_setindex(J_cache)
        copyto!(J_cache, J)
        if fast_scalar_indexing(J_cache)
            @inbounds for i in axes(J_cache, 1)
                J_cache[i, i] += D[i, i]
            end
        else
            idxs = diagind(J_cache)
            @.. broadcast=false @view(J_cache[idxs])=@view(J[idxs]) + @view(D[idxs])
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
            @inbounds for i in axes(J_cache, 1)
                J_cache[i, i] += D
            end
        else
            idxs = diagind(J_cache)
            @.. broadcast=false @view(J_cache[idxs])=@view(J[idxs]) + D
        end
        return J_cache
    else
        return @. J + D
    end
end
