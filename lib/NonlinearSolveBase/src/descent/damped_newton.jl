"""
    DampedNewtonDescent(; linsolve = nothing, initial_damping, damping_fn,
                          min_norm_mode = :auto)

A Newton descent algorithm with damping. The damping factor is computed using the
`damping_fn` function. The descent direction is computed as ``(JᵀJ + λDᵀD) δu = -fu``. For
non-square Jacobians, we default to solving for `Jδx = -fu` and `√λ⋅D δx = 0`
simultaneously. If the linear solver can't handle non-square matrices, we use the normal
form equations ``(JᵀJ + λDᵀD) δu = Jᵀ fu``. Note that this factorization is often the faster
choice, but it is not as numerically stable as the least squares solver.

For underdetermined systems (more unknowns than equations, `length(u) > length(fu)`) with
a normal form damping function, we default to a minimum-norm formulation: solve
``(JJᵀ + λD̃ᵀD̃) z = -fu`` and set ``δu = Jᵀz``, where `D̃` is the damping for the `JJᵀ`
system. As ``λ → 0`` this recovers the minimum-norm step solving the linearized equations,
and it keeps the linear system small (`m × m` where `m = length(fu)` instead of
`n × n`).

The damping factor returned must be a non-negative number.

### Keyword Arguments

  - `initial_damping`: the initial damping factor to use
  - `damping_fn`: the function to use to compute the damping factor. This must satisfy the
    [`NonlinearSolveBase.AbstractDampingFunction`](@ref) interface.
  - `min_norm_mode`: controls the minimum-norm formulation for underdetermined systems:

      + `:auto` (default): use the minimum-norm formulation for underdetermined systems
        when the damping function returns normal form damping
      + `:minimum_norm`: force the minimum-norm formulation regardless of system dimensions
      + `:disabled`: never use the minimum-norm formulation
"""
@kwdef @concrete struct DampedNewtonDescent <: AbstractDescentDirection
    linsolve = nothing
    initial_damping
    damping_fn <: AbstractDampingFunction
    min_norm_mode::Symbol = :auto
end

supports_line_search(::DampedNewtonDescent) = true
supports_trust_region(::DampedNewtonDescent) = true

@concrete mutable struct DampedNewtonDescentCache <: AbstractDescentCache
    J
    δu
    δus
    lincache
    JᵀJ_cache
    Jᵀfu_cache
    rhs_cache
    damping_fn_cache
    timer
    preinverted_jacobian <: Union{Val{false}, Val{true}}
    mode <: Union{Val{:normal_form}, Val{:least_squares}, Val{:simple}, Val{:minimum_norm}}
    z_cache    # only used in `:minimum_norm` mode, solution of the m×m dual system
end

@internal_caches DampedNewtonDescentCache :lincache :damping_fn_cache

function InternalAPI.init(
        prob::AbstractNonlinearProblem, alg::DampedNewtonDescent, J, fu, u; stats,
        pre_inverted::Val = Val(false), linsolve_kwargs = (;),
        abstol = nothing, reltol = nothing,
        timer = get_timer_output(),
        alias_J::Bool = true, shared::Val = Val(1),
        kwargs...
    )
    length(fu) != length(u) &&
        @assert pre_inverted isa Val{false} "Precomputed Inverse for Non-Square Jacobian doesn't make sense."

    @bb δu = similar(u)
    δus = Utils.unwrap_val(shared) ≤ 1 ? nothing : map(2:Utils.unwrap_val(shared)) do i
            @bb δu_ = similar(u)
    end

    normal_form_damping = returns_norm_form_damping(alg.damping_fn)
    normal_form_linsolve = needs_square_A(alg.linsolve, u)

    alg.min_norm_mode in (:auto, :minimum_norm, :disabled) ||
        throw(ArgumentError("`min_norm_mode` must be `:auto`, `:minimum_norm`, or \
                             `:disabled`, got `$(alg.min_norm_mode)`."))
    use_minimum_norm = alg.min_norm_mode === :minimum_norm ||
        (
        alg.min_norm_mode === :auto && length(fu) < length(u) &&
            normal_form_damping
    )
    if use_minimum_norm && !normal_form_damping
        throw(ArgumentError("`min_norm_mode = :minimum_norm` requires a damping function \
                             that returns normal form damping."))
    end

    mode = if u isa Number
        :simple
    elseif use_minimum_norm
        :minimum_norm
    elseif prob isa NonlinearProblem
        if normal_form_damping
            ifelse(normal_form_linsolve, :normal_form, :least_squares)
        else
            :simple
        end
    else
        if normal_form_linsolve & !normal_form_damping
            throw(ArgumentError("Linear Solver expects Normal Form but returned Damping is \
                                 not Normal Form. This is not supported."))
        end
        if normal_form_damping & !normal_form_linsolve
            :least_squares
        else
            ifelse(!normal_form_damping & !normal_form_linsolve, :simple, :normal_form)
        end
    end

    z_cache = nothing

    if mode === :minimum_norm
        # The minimum-norm formulation is the normal form of the dual system: solve
        # (JJᵀ + λD̃ᵀD̃) z = -fu for the m-sized z, then δu = Jᵀz. The damping function
        # is initialized on the dual system, so `fu` takes the place of `u` (and of the
        # normal form rhs Jᵀfu) and `JJᵀ` the place of `JᵀJ`.
        JᵀJ = J * transpose(J)
        jac_damp = requires_normal_form_jacobian(alg.damping_fn) ? JᵀJ : J

        damping_fn_cache = InternalAPI.init(
            prob, alg.damping_fn, alg.initial_damping, jac_damp, fu, fu, Val(true);
            stats, kwargs...
        )
        D = damping_fn_cache(nothing)

        if ArrayInterface.can_setindex(JᵀJ)
            @bb J_cache = similar(JᵀJ)
            @bb @. J_cache = 0
        else
            J_cache = JᵀJ
        end
        J_damped = dampen_jacobian!!(J_cache, JᵀJ, D)
        @bb z_cache = similar(fu)

        A, b = Utils.maybe_symmetric(J_damped), Utils.safe_vec(fu)
        Jᵀfu, rhs_cache = nothing, nothing
    elseif mode === :least_squares
        if requires_normal_form_jacobian(alg.damping_fn)
            JᵀJ = transpose(J) * J  # Needed to compute the damping factor
            jac_damp = JᵀJ
        else
            JᵀJ = nothing
            jac_damp = J
        end
        if requires_normal_form_rhs(alg.damping_fn)
            Jᵀfu = transpose(J) * Utils.safe_vec(fu)
            rhs_damp = Jᵀfu
        else
            Jᵀfu = nothing
            rhs_damp = fu
        end

        damping_fn_cache = InternalAPI.init(
            prob, alg.damping_fn, alg.initial_damping, jac_damp, rhs_damp, u, Val(false);
            stats, kwargs...
        )
        D = damping_fn_cache(nothing)

        D isa Number && (D = D * LinearAlgebra.I)
        rhs_cache = vcat(Utils.safe_vec(fu), Utils.safe_vec(u))
        J_cache = Utils.faster_vcat(J, D)
        A, b = J_cache, rhs_cache
    elseif mode === :simple
        damping_fn_cache = InternalAPI.init(
            prob, alg.damping_fn, alg.initial_damping, J, fu, u, Val(false); kwargs...
        )
        J_cache = Utils.maybe_unaliased(J, alias_J)
        D = damping_fn_cache(nothing)

        J_damped = dampen_jacobian!!(J_cache, J, D)
        J_cache = J_damped
        A, b = J_damped, Utils.safe_vec(fu)
        JᵀJ, Jᵀfu, rhs_cache = nothing, nothing, nothing
    elseif mode === :normal_form
        JᵀJ = transpose(J) * J
        Jᵀfu = transpose(J) * Utils.safe_vec(fu)
        jac_damp = requires_normal_form_jacobian(alg.damping_fn) ? JᵀJ : J
        rhs_damp = requires_normal_form_rhs(alg.damping_fn) ? Jᵀfu : fu

        damping_fn_cache = InternalAPI.init(
            prob, alg.damping_fn, alg.initial_damping, jac_damp, rhs_damp, u, Val(true);
            stats, kwargs...
        )
        D = damping_fn_cache(nothing)

        @bb J_cache = similar(JᵀJ)
        @bb @. J_cache = 0
        J_damped = dampen_jacobian!!(J_cache, JᵀJ, D)
        A, b = Utils.maybe_symmetric(J_damped), Utils.safe_vec(Jᵀfu)
        rhs_cache = nothing
    end

    linsolve_u = mode === :minimum_norm ? Utils.safe_vec(z_cache) : Utils.safe_vec(u)
    lincache = construct_linear_solver(
        alg, alg.linsolve, A, b, linsolve_u, prob.p;
        stats, abstol, reltol, linsolve_kwargs...
    )

    return DampedNewtonDescentCache(
        J_cache, δu, δus, lincache, JᵀJ, Jᵀfu, rhs_cache,
        damping_fn_cache, timer, pre_inverted, Val(mode), z_cache
    )
end

function InternalAPI.solve!(
        cache::DampedNewtonDescentCache, J, fu, u, idx::Val = Val(1);
        skip_solve::Bool = false, new_jacobian::Bool = true, kwargs...
    )
    δu = SciMLBase.get_du(cache, idx)
    skip_solve && return DescentResult(; δu)

    recompute_A = idx === Val(1)

    @static_timeit cache.timer "dampen" begin
        if cache.mode isa Val{:minimum_norm}
            if (J !== nothing || new_jacobian) && recompute_A
                preinverted_jacobian(cache) && (J = inv(J))
                @bb cache.JᵀJ_cache = J × transpose(J)
                D = InternalAPI.solve!(cache.damping_fn_cache, cache.JᵀJ_cache, fu, Val(true))
                cache.J = dampen_jacobian!!(cache.J, cache.JᵀJ_cache, D)
                A = Utils.maybe_symmetric(cache.J)
            elseif !recompute_A
                A = Utils.maybe_symmetric(cache.J)
            else
                A = nothing
            end
            b = Utils.safe_vec(fu)
        elseif cache.mode isa Val{:least_squares}
            if (J !== nothing || new_jacobian) && recompute_A
                preinverted_jacobian(cache) && (J = inv(J))
                if requires_normal_form_jacobian(cache.damping_fn_cache)
                    @bb cache.JᵀJ_cache = transpose(J) × J
                    jac_damp = cache.JᵀJ_cache
                else
                    jac_damp = J
                end
                if requires_normal_form_rhs(cache.damping_fn_cache)
                    @bb cache.Jᵀfu_cache = transpose(J) × vec(fu)
                    rhs_damp = cache.Jᵀfu_cache
                else
                    rhs_damp = fu
                end
                D = InternalAPI.solve!(
                    cache.damping_fn_cache, jac_damp, rhs_damp, Val(false)
                )
                if Utils.can_setindex(cache.J)
                    copyto!(@view(cache.J[1:size(J, 1), :]), J)
                    cache.J[(size(J, 1) + 1):end, :] .= sqrt.(D)
                else
                    cache.J = Utils.faster_vcat(J, sqrt.(D))
                end
            end
            A = cache.J
            if Utils.can_setindex(cache.rhs_cache)
                cache.rhs_cache[1:length(fu)] .= Utils.safe_vec(fu)
                cache.rhs_cache[(length(fu) + 1):end] .= false
            else
                cache.rhs_cache = vcat(Utils.safe_vec(fu), zero(Utils.safe_vec(u)))
            end
            b = cache.rhs_cache
        elseif cache.mode isa Val{:simple}
            if (J !== nothing || new_jacobian) && recompute_A
                preinverted_jacobian(cache) && (J = inv(J))
                D = InternalAPI.solve!(cache.damping_fn_cache, J, fu, Val(false))
                cache.J = dampen_jacobian!!(cache.J, J, D)
            end
            A, b = cache.J, Utils.safe_vec(fu)
        elseif cache.mode isa Val{:normal_form}
            if (J !== nothing || new_jacobian) && recompute_A
                preinverted_jacobian(cache) && (J = inv(J))
                @bb cache.JᵀJ_cache = transpose(J) × J
                @bb cache.Jᵀfu_cache = transpose(J) × vec(fu)
                D = InternalAPI.solve!(
                    cache.damping_fn_cache, cache.JᵀJ_cache, cache.Jᵀfu_cache, Val(true)
                )
                cache.J = dampen_jacobian!!(cache.J, cache.JᵀJ_cache, D)
                A = Utils.maybe_symmetric(cache.J)
            elseif !recompute_A
                @bb cache.Jᵀfu_cache = transpose(J) × vec(fu)
                A = Utils.maybe_symmetric(cache.J)
            else
                A = nothing
            end
            b = cache.Jᵀfu_cache
        else
            error(lazy"Unknown Mode: $(cache.mode).")
        end
    end

    @static_timeit cache.timer "linear solve" begin
        min_norm = cache.mode isa Val{:minimum_norm}
        linres = cache.lincache(;
            A, b,
            reuse_A_if_factorization = !new_jacobian && !recompute_A,
            kwargs...,
            linu = min_norm ? Utils.safe_vec(cache.z_cache) : Utils.safe_vec(δu)
        )
        if min_norm
            z = Utils.restructure(cache.z_cache, linres.u)
            @bb δu = transpose(J) × vec(z)
        end
        δu = Utils.restructure(SciMLBase.get_du(cache, idx), min_norm ? δu : linres.u)
        if !linres.success
            set_du!(cache, δu, idx)
            return DescentResult(; δu, success = false, linsolve_success = false)
        end
    end

    @bb @. δu *= -1
    set_du!(cache, δu, idx)
    return DescentResult(; δu)
end

dampen_jacobian!!(::Any, J::Union{AbstractSciMLOperator, Number}, D) = J + D
function dampen_jacobian!!(J_cache, J::AbstractMatrix, D::Union{AbstractMatrix, Number})
    ArrayInterface.can_setindex(J_cache) || return J .+ D
    J_cache !== J && copyto!(J_cache, J)
    if ArrayInterface.fast_scalar_indexing(J_cache)
        if D isa Number
            @simd ivdep for i in axes(J_cache, 1)
                @inbounds J_cache[i, i] += D
            end
        else
            @simd ivdep for i in axes(J_cache, 1)
                @inbounds J_cache[i, i] += D[i, i]
            end
        end
    else
        idxs = diagind(J_cache)
        if D isa Number
            J_cache[idxs] .+= D
        else
            J_cache[idxs] .+= @view(D[idxs])
        end
    end
    return J_cache
end
