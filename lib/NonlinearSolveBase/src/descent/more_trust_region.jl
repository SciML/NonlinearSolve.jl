"""
    MoreTrustRegionDescent(; linsolve = nothing, scaling = :none, min_damping_D = 1e-8)

Compute the descent direction by solving the trust-region subproblem
`min ‖J δu + fu‖` subject to `‖D δu‖ ≤ Δ` nearly exactly. A safeguarded Newton
iteration finds the damping parameter `λ` such that the least-squares problem

    min ‖ [J; √λ D] δu - [-fu; 0] ‖

— equivalent to the damped normal equations `(JᵀJ + λDᵀD) δu = -Jᵀfu` — has
solution with `‖D δu‖ = Δ`; if the Gauss-Newton step already lies inside the
region it is the subproblem solution and `λ = 0` is returned directly. This is
the algorithm of Moré (MINPACK `lmpar`), also described in Nocedal & Wright
§10.3.

Solving the augmented least-squares system keeps the condition number of `J`
unsquared and handles rank-deficient Jacobians natively: `[J; √λ D]` has full
column rank for every `λ > 0`. When the chosen `linsolve` only supports square
systems the equivalent damped normal equations are used instead, and for
matrix-free (operator) Jacobians the augmented system is applied as a stacked
operator, which Krylov least-squares solvers handle without forming `J`.

Unlike [`Dogleg`](@ref), which follows a two-piece polygonal approximation of
the solution curve, this descent follows the true solution `δu(λ)`, which is
substantially more robust on ill-conditioned least-squares problems. Pair it
with `RadiusUpdateSchemes.More` to get Moré's step-following radius update as
well.

### Keyword Arguments

  - `linsolve`: the linear solver used for the subproblem solves. The default
    path solves the rectangular augmented system, so `linsolve` should handle
    least-squares problems (the default choice does); solvers requiring square
    systems such as `LUFactorization` are routed through the normal equations
    automatically.
  - `scaling`: the diagonal scaling matrix `D`. `:none` uses `D = I`;
    `:jacobian` uses Moré's scaling `Dᵢᵢ = max(Dᵢᵢ, ‖J[:, i]‖)`, which never
    decreases across iterations and makes the trust region scale-covariant.
    `:jacobian` requires a concrete Jacobian.
  - `min_damping_D`: lower bound for the entries of `DᵀD` under `:jacobian`
    scaling.
"""
@kwdef @concrete struct MoreTrustRegionDescent <: AbstractDescentDirection
    linsolve = nothing
    scaling::Symbol = :none
    min_damping_D = 1.0e-8
end

supports_trust_region(::MoreTrustRegionDescent) = true

# `normal_form` selects the subproblem formulation: `true` solves the damped
# normal equations `(JᵀJ + λDᵀD) p = -Jᵀfu` (scalars, and `linsolve`s that only
# accept square systems), `false` solves the augmented least-squares system
# `[J; √λD] p = [-fu; 0]` directly (concrete matrices through `linsolve`, and
# matrix-free Jacobians through a stacked operator).
@concrete mutable struct MoreTrustRegionDescentCache <: AbstractDescentCache
    δu
    δus
    lincache    # solver for the damped/augmented system and the Gauss-Newton step
    Jᵀfu        # `Jᵀ fu` for the primary direction
    JᵀJ         # normal-form only
    damped      # normal-form in-place buffer for `JᵀJ + λD²`, else `nothing`
    augmented   # `[J; √λD]` buffer for mutable dense `J`, else `nothing`
    rhs         # `[-fu; 0]` buffer, else `nothing`
    qrhs        # `[0; Dp/√λ]` buffer, else `nothing`
    p           # current trial step
    gn_step     # Gauss-Newton step, reused while `J` and `fu` are unchanged
    gn_norm     # scaled norm `‖D δu_gn‖`; `Inf` when the GN solve failed
    gn_valid::Bool
    # pinned field type: `alg.scaling` is a runtime `Symbol`, so an inferred
    # `Union` param here would split the cache type across `:none`/`:jacobian`
    dtd::Union{Nothing, AbstractVector, Number}
    Dp
    D²p
    q
    Jδu         # `J δu` product buffer for the predicted reduction
    λ
    θ
    maxiters::Int
    min_damping_D
    internalnorm
    timer
    preinverted_jacobian <: Union{Val{false}, Val{true}}
    normal_form <: Union{Val{false}, Val{true}}
    jac_convert <: Union{Val{false}, Val{true}}
    op_state
end

_more_jac_convert(cache::MoreTrustRegionDescentCache) =
    Utils.unwrap_val(cache.jac_convert)

# Operator Jacobians keep a single `FunctionOperator` whose closures read `J`/`λ`
# through this cell: `set_lincache_A!` would `copyto!` a rebuilt operator
# (`FunctionOperator` reports `can_setindex` but defines no `copyto!`), so `A` is
# set once at init — `alias_A = true` keeps it — and the cell carries the updates.
mutable struct _MoreOpState{T}
    J::Any
    λ::T
    Jv::Vector{T}
end

# `[J; √λ I]` as an `(m + n) × n` operator: forward `x ↦ [Jx; √λ x]`, adjoint
# `[y₁; y₂] ↦ Jᵀy₁ + √λ y₂`. `:jacobian` scaling is rejected for matrix-free
# Jacobians, so the damping block is always √λ I here.
function _more_augmented_operator(state::_MoreOpState, u, fu)
    m, n = length(fu), length(u)
    T = promote_type(eltype(u), eltype(fu))
    return FunctionOperator(
        (w, x, _, p, t) -> begin
            mul!(@view(w[1:m]), state.J, x)
            @views @. w[(m + 1):(m + n)] = sqrt(state.λ) * x
            return w
        end,
        Vector{T}(undef, n), Vector{T}(undef, m + n);
        op_adjoint = (w, y, _, p, t) -> begin
            mul!(w, state.J', @view(y[1:m]))
            @views @. w += sqrt(state.λ) * y[(m + 1):(m + n)]
            return w
        end,
        isconstant = true, islinear = true
    )
end

# `x ↦ (JᵀJ + λI)x` for square-only Krylov solvers; `state.Jv` is matvec scratch.
# Symmetric positive definite for λ > 0, so CG/MINRES apply too.
function _more_normal_form_operator(state::_MoreOpState{T}, u) where {T}
    n = length(u)
    nf_op! = (w, x, _, p, t) -> begin
        mul!(state.Jv, state.J, x)
        mul!(w, state.J', state.Jv)
        @. w += state.λ * x
        return w
    end
    return FunctionOperator(
        nf_op!, Vector{T}(undef, n), Vector{T}(undef, n);
        op_adjoint = nf_op!, isconstant = true, islinear = true,
        issymmetric = true, ishermitian = true, isposdef = true
    )
end

function InternalAPI.init(
        prob::AbstractNonlinearProblem, alg::MoreTrustRegionDescent, J, fu, u; stats,
        pre_inverted::Val = Val(false), linsolve_kwargs = (;),
        abstol = nothing, reltol = nothing, internalnorm::F = L2_NORM,
        shared::Val = Val(1), timer = get_timer_output(), kwargs...
    ) where {F}
    length(fu) != length(u) &&
        @assert !Utils.unwrap_val(pre_inverted) "Precomputed Inverse for Non-Square Jacobian doesn't make sense."
    alg.scaling in (:none, :jacobian) ||
        throw(ArgumentError("`scaling` must be `:none` or `:jacobian`, got \
                             `$(alg.scaling)`."))
    isop = J === nothing || J isa AbstractSciMLOperator
    if isop
        Utils.unwrap_val(pre_inverted) &&
            throw(ArgumentError("`MoreTrustRegionDescent` cannot invert a matrix-free \
                                 Jacobian; use `concrete_jac = true` or a different \
                                 descent algorithm."))
        alg.scaling === :jacobian &&
            throw(ArgumentError("`scaling = :jacobian` needs column norms of a \
                                 concrete Jacobian; use `scaling = :none` for \
                                 matrix-free problems."))
        if needs_concrete_A(alg.linsolve)
            # a convertible operator (`MatrixOperator`-style `jac_prototype`) is
            # materialized once, like factorization solvers do lazily; a genuinely
            # matrix-free Jacobian cannot feed a concrete-only solver
            (J isa AbstractSciMLOperator && isconvertible(J)) ||
                throw(ArgumentError("`MoreTrustRegionDescent` on a matrix-free Jacobian \
                                     needs a `linsolve` that acts on operators (the \
                                     default Krylov selection is one); \
                                     `$(alg.linsolve)` requires a concrete system."))
        end
    end
    jac_convert = isop && needs_concrete_A(alg.linsolve)

    J_ = Utils.unwrap_val(pre_inverted) ? inv(J) : J
    jac_convert && (J_ = convert(AbstractMatrix, J_); isop = false)
    T = promote_type(eltype(u), eltype(fu))

    # Statics go through the normal equations as well: `SMatrix \ vector` on a
    # rectangular augmented system hits a missing `adjoint` in StaticArrays' QR,
    # while the small square `JᵀJ` is cheap to form. Operator Jacobians take the
    # normal-form path when `linsolve` needs a square system: `JᵀJ + λI` is then
    # applied as an operator.
    normal_form = u isa Number || J isa Number || J_ isa StaticArray ||
        needs_square_A(alg.linsolve, u)

    @bb δu = zero(u)
    δus = Utils.unwrap_val(shared) ≤ 1 ? nothing : map(2:Utils.unwrap_val(shared)) do i
            @bb δu_ = zero(u)
    end

    if u isa Number
        p, gn_step, Dp, D²p, q = zero(u), zero(u), zero(u), zero(u), zero(u)
        Jᵀfu, Jδu, augmented, rhs, qrhs = zero(u), zero(fu), nothing, nothing, nothing
    else
        # `p`, `Jᵀfu`, ... live in the vectorized linear-algebra space: for a
        # matrix-shaped `u`/`fu` they are `length(u)`/`length(fu)` vectors while `δu`
        # keeps the state shape (`restructure` maps between them at the boundary)
        @bb p = similar(Utils.safe_vec(u))
        @bb gn_step = similar(Utils.safe_vec(u))
        @bb Dp = similar(Utils.safe_vec(u))
        @bb D²p = similar(Utils.safe_vec(u))
        @bb q = similar(Utils.safe_vec(u))
        @bb Jᵀfu = similar(Utils.safe_vec(u))
        Jδu = if fu isa Number
            zero(fu)
        else
            @bb Jδu_ = similar(Utils.safe_vec(fu))
            Jδu_
        end
        if normal_form
            augmented = rhs = qrhs = nothing
        else
            m, n = length(fu), length(u)
            augmented = if J_ isa AbstractMatrix &&
                    ArrayInterface.can_setindex(J_) &&
                    ArrayInterface.fast_scalar_indexing(J_)
                zeros(T, m + n, n)
            else
                nothing
            end
            rhs = zeros(T, m + n)
            qrhs = zeros(T, m + n)
        end
    end

    op_state = isop ? _MoreOpState(J_, zero(T), Vector{T}(undef, length(u))) : nothing
    if normal_form
        JᵀJ = if isop
            nothing
        elseif J_ isa Number
            abs2(J_)
        else
            transpose(J_) * J_
        end
        dtd = alg.scaling === :jacobian ?
            _more_scaling_init(JᵀJ, u, alg.min_damping_D) : nothing
        damped = if JᵀJ isa AbstractMatrix && ArrayInterface.can_setindex(JᵀJ) &&
                ArrayInterface.fast_scalar_indexing(JᵀJ)
            @bb damped = similar(JᵀJ)
            damped
        else
            nothing
        end
        A0 = if isop
            _more_normal_form_operator(op_state, u)
        else
            A0 = _more_damped_system(JᵀJ, T(1.0e-3), dtd, damped, u)
            A0 isa AbstractMatrix ? Utils.maybe_symmetric(A0) : A0
        end
        Jᵀfu0 = J_ isa Number ? J_ * fu : transpose(J_) * Utils.safe_vec(fu)
        linsolve = alg.linsolve === nothing ? default_spd_linsolve(A0) : alg.linsolve
        lincache = construct_linear_solver(
            alg, linsolve, A0, Utils.safe_vec(Jᵀfu0), Utils.safe_vec(u), prob.p;
            stats, abstol, reltol, linsolve_kwargs...
        )
    else
        JᵀJ = damped = nothing
        dtd = alg.scaling === :jacobian ?
            _more_scaling_init(J_, u, alg.min_damping_D, Val(:columns)) : nothing
        A0 = if isop
            _more_augmented_operator(op_state, u, fu)
        else
            _more_augmented_system(J_, T(1.0e-3), dtd, augmented, fu, u)
        end
        b0 = _more_augmented_rhs!(rhs, fu)
        lincache = construct_linear_solver(
            alg, alg.linsolve, A0, b0, Utils.safe_vec(u), prob.p;
            stats, abstol, reltol, linsolve_kwargs...
        )
    end

    return MoreTrustRegionDescentCache(
        δu, δus, lincache, Jᵀfu, JᵀJ, damped, augmented, rhs, qrhs,
        p, gn_step, T(Inf), false, dtd, Dp, D²p, q, Jδu, zero(T),
        T(1.0e-4), 10, T(alg.min_damping_D), internalnorm, timer, pre_inverted,
        Val(normal_form), Val(jac_convert), op_state
    )
end

function InternalAPI.reinit!(cache::MoreTrustRegionDescentCache, args...; kwargs...)
    InternalAPI.reinit!(cache.lincache, args...; kwargs...)
    cache.λ = zero(cache.λ)
    cache.gn_valid = false
    cache.dtd isa AbstractVector && fill!(cache.dtd, cache.min_damping_D)
    cache.dtd isa Number && (cache.dtd = cache.min_damping_D)
    return
end

function NonlinearSolveBase.callback_into_cache!(
        topcache, cache::MoreTrustRegionDescentCache, args...
    )
    # An accepted step changed `fu`, so the cached Gauss-Newton step and `Jᵀfu` are stale
    # even when the Jacobian was reused
    tr_cache = Utils.safe_getproperty(topcache, Val(:trustregion_cache))
    tr_cache isa AbstractTrustRegionMethodCache &&
        NonlinearSolveBase.last_step_accepted(tr_cache) && (cache.gn_valid = false)
    return NonlinearSolveBase.callback_into_cache!(cache, cache.lincache)
end

# `dtd` stores the squared diagonal of `D`; `D δu` and `D² δu` go through these helpers
function _more_Dp!(cache, dtd::AbstractVector, p)
    @bb @. cache.Dp = sqrt(dtd) * p
    return cache.Dp
end
_more_Dp!(cache, ::Nothing, p) = p
_more_Dp!(cache, dtd::Number, p::Number) = sqrt(dtd) * p
function _more_D2p!(cache, dtd::AbstractVector, p)
    @bb @. cache.D²p = dtd * p
    return cache.D²p
end
_more_D2p!(cache, ::Nothing, p) = p
_more_D2p!(cache, dtd::Number, p::Number) = dtd * p

_more_scaled_norm(cache, p) = cache.internalnorm(_more_Dp!(cache, cache.dtd, p))
function _more_scaled_norm(cache, p::Number)
    cache.dtd === nothing && return abs(p)
    return sqrt(cache.dtd) * abs(p)
end

# `Dᵢᵢ = max(Dᵢᵢ, ‖J[:, i]‖)`: read off the `JᵀJ` diagonal under normal form, the
# column norms of `J` directly otherwise — the same numbers either way
function _more_scaling_init(JᵀJ::AbstractMatrix, u, min_damping)
    dtd = _more_dtd_buffer(u)
    @inbounds for i in axes(JᵀJ, 1)
        dtd[i] = max(abs(JᵀJ[i, i]), min_damping)
    end
    return dtd
end
function _more_scaling_init(J::AbstractMatrix, u, min_damping, ::Val{:columns})
    dtd = _more_dtd_buffer(u)
    @inbounds for j in axes(J, 2)
        dtd[j] = max(abs2(norm(@view(J[:, j]))), min_damping)
    end
    return dtd
end
_more_scaling_init(JᵀJ::Number, u, min_damping) = max(JᵀJ, min_damping)

# `similar(u)` keeps SVector-shaped states on a static diagonal; everything mutable
# goes through the usual concrete vector
function _more_dtd_buffer(u)
    dtd = similar(vec(u))
    ArrayInterface.can_setindex(dtd) ||
        (dtd = Vector{eltype(dtd)}(undef, length(dtd)))
    return dtd
end

function _more_scaling_update!(cache, JᵀJ::AbstractMatrix)
    cache.dtd === nothing && return
    @inbounds for i in axes(JᵀJ, 1)
        cache.dtd[i] = max(cache.dtd[i], abs(JᵀJ[i, i]), cache.min_damping_D)
    end
    return
end
function _more_scaling_update!(cache, J::AbstractMatrix, ::Val{:columns})
    cache.dtd === nothing && return
    @inbounds for j in axes(J, 2)
        cache.dtd[j] = max(
            cache.dtd[j], abs2(norm(@view(J[:, j]))), cache.min_damping_D
        )
    end
    return
end
# `:jacobian` scaling is rejected for operator Jacobians at init, so `dtd === nothing`
_more_scaling_update!(cache, ::Any, ::Val{:columns}) = nothing
_more_scaling_update!(cache, ::AbstractSciMLOperator) = nothing
function _more_scaling_update!(cache, JᵀJ::Number)
    cache.dtd === nothing && return
    cache.dtd = max(cache.dtd, abs(JᵀJ), cache.min_damping_D)
    return
end

function _more_damped_system(JᵀJ::Number, λ, dtd, damped, u)
    return JᵀJ + λ * (dtd === nothing ? one(JᵀJ) : dtd)
end
function _more_damped_system(JᵀJ::AbstractMatrix, λ, dtd, damped, u)
    if damped === nothing
        return JᵀJ + λ * (dtd === nothing ? LinearAlgebra.I : Diagonal(dtd))
    end
    copyto!(damped, JᵀJ)
    @simd ivdep for i in axes(damped, 1)
        @inbounds damped[i, i] += λ * (dtd === nothing ? one(λ) : dtd[i])
    end
    return damped
end
# `[J; √λ D]` assembled into `buf` for mutable dense `J` (rewritten in full each
# λ since the factorization may consume it); `vcat` keeps sparse `J` sparse
function _more_augmented_system(J_::AbstractMatrix, λ, dtd, buf, fu, u)
    if buf === nothing
        return vcat(J_, _more_damping_block(J_, λ, dtd))
    end
    m, n = size(J_)
    copyto!(@view(buf[1:m, :]), J_)
    fill!(@view(buf[(m + 1):(m + n), :]), zero(eltype(buf)))
    @inbounds for j in 1:n
        buf[m + j, j] = sqrt(λ) * (dtd === nothing ? one(λ) : sqrt(dtd[j]))
    end
    return buf
end

# `√λ D` as an `n × n` bottom block matching `J`'s storage flavor
# `vcat` promotes `Diagonal` to sparse under a sparse `J`, so only the static
# case needs an explicit `SMatrix` block to keep the system fully static
function _more_damping_block(J_::StaticArray, λ, dtd)
    n = size(J_, 2)
    T = promote_type(eltype(J_), typeof(λ))
    return dtd === nothing ? sqrt(λ) * SMatrix{n, n, T}(LinearAlgebra.I) :
        SMatrix{n, n, T}(Diagonal(SVector{n, T}(sqrt.(λ .* dtd))))
end
function _more_damping_block(J_::AbstractMatrix, λ, dtd)
    n = size(J_, 2)
    T = promote_type(eltype(J_), typeof(λ))
    d = dtd === nothing ? ones(T, n) : sqrt.(dtd)
    return sqrt(λ) * Diagonal(d)
end

function _more_augmented_rhs!(rhs, fu)
    m = length(fu)
    @views rhs[1:m] .= .-Utils.safe_vec(fu)
    fill!(@view(rhs[(m + 1):end]), zero(eltype(rhs)))
    return rhs
end

function _more_augmented_qrhs!(qrhs, Dp, λ, m)
    fill!(@view(qrhs[1:m]), zero(eltype(qrhs)))
    @views @. qrhs[(m + 1):end] = Dp / sqrt(λ)
    return qrhs
end

_all_finite(x::Number) = isfinite(x)
_all_finite(x) = all(isfinite, x)

# `eps * maxdiag(JᵀJ) / min(diag D²)`: below this the damped system cannot factorize as
# positive definite, so the safeguarded iteration must not start or retry below it
function _more_maxdiag(JᵀJ::AbstractMatrix)
    m = zero(eltype(JᵀJ))
    @inbounds for i in axes(JᵀJ, 1)
        m = max(m, abs(JᵀJ[i, i]))
    end
    return m
end
_more_maxdiag(JᵀJ::Number) = abs(JᵀJ)
# `JᵀJ` is `nothing` for operator Jacobians: the λ floor is only needed against
# factorization failure, which Krylov solves do not hit
_more_maxdiag(::Union{Nothing, AbstractSciMLOperator}) = true
_more_mindtd(::Nothing) = 1
_more_mindtd(dtd::AbstractVector) = minimum(dtd)
_more_mindtd(dtd::Number) = dtd

# Moré's safeguarded Newton update on φ(λ) = ‖D δu‖ - Δ: tighten the bracket [l, u] on
# λ* and fall back to a guarded step if the Newton iterate leaves it
function _more_update_λ(λ, ϕ, Δ, pᵀD²p, pᵀD²q, l, u)
    ϕ < 0 ? (u = λ) : (l = λ)
    λ += ϕ / Δ * pᵀD²p / pᵀD²q
    l <= λ <= u || (λ = max(l + 0.01 * (u - l), sqrt(l * u)))
    return λ, l, u
end

function InternalAPI.solve!(
        cache::MoreTrustRegionDescentCache, J, fu, u, idx::Val = Val(1);
        skip_solve::Bool = false, new_jacobian::Bool = true,
        trust_region = nothing, kwargs...
    )
    δu = SciMLBase.get_du(cache, idx)
    skip_solve && return DescentResult(; δu)
    @assert trust_region !== nothing "`trust_region` must be specified for \
        `MoreTrustRegionDescent`."

    T = promote_type(eltype(u), eltype(fu))
    Δ = T(trust_region)
    idx1 = idx === Val(1)
    J_ = preinverted_jacobian(cache) ? inv(J) : J
    _more_jac_convert(cache) && (J_ = convert(AbstractMatrix, J_))
    cache.op_state !== nothing && (cache.op_state.J = J_)

    if new_jacobian && idx1
        if normal_form(cache) && cache.op_state === nothing
            if J_ isa Number
                cache.JᵀJ = abs2(J_)
            else
                @bb cache.JᵀJ = transpose(J_) × J_
            end
            _more_scaling_update!(cache, cache.JᵀJ)
        else
            _more_scaling_update!(cache, J_, Val(:columns))
        end
        cache.gn_valid = false
    end

    # Gauss-Newton step: when it lies inside the region it is the subproblem solution.
    # On a rejected retry (`new_jacobian = false`) both `J` and `fu` are unchanged, so
    # the cached step is reused without an extra factorization. Secondary directions
    # (`idx > 1`, e.g. under `GeodesicAcceleration`) evaluate at a perturbed `fu`, so
    # they recompute `Jᵀfu` and the GN step into scratch without touching the cache.
    Jᵀfu, gn_step, gn_norm = if idx1 && cache.gn_valid
        (cache.Jᵀfu, cache.gn_step, cache.gn_norm)
    else
        Jᵀfu = if idx1
            if J_ isa Number
                cache.Jᵀfu = J_ * fu
            else
                @bb cache.Jᵀfu = transpose(J_) × Utils.safe_vec(fu)
            end
        else
            J_ isa Number ? J_ * fu : transpose(J_) * Utils.safe_vec(fu)
        end
        # Jᵀfu = 0 is a stationary point of the model: p = 0 solves the subproblem for
        # every Δ, and the λ bracket below degenerates to the empty interval (0, 0].
        if iszero(cache.internalnorm(Jᵀfu))
            δu = Utils.restructure(δu, zero(cache.p))
            set_du!(cache, δu, idx)
            extras = _more_extras(cache, J_, δu, zero(T))
            return DescentResult(; δu, extras)
        end
        gn_buf = idx1 ? cache.gn_step : cache.p
        linres = _more_gn_solve(cache, J_, Jᵀfu, fu, gn_buf, u, kwargs)
        if linres.success && _all_finite(linres.u)
            if gn_buf isa AbstractArray && ArrayInterface.can_setindex(gn_buf)
                if normal_form(cache)
                    @bb @. gn_buf = -linres.u
                else
                    @bb @. gn_buf = linres.u
                end
                gn = gn_buf
            else
                gn = normal_form(cache) ?
                    (linres.u isa Number ? -linres.u : .-linres.u) : linres.u
                gn = Utils.restructure(gn_buf, gn)
            end
            idx1 && (cache.gn_step = gn)
            gn_norm = _more_scaled_norm(cache, gn)
        else
            gn = nothing
            gn_norm = T(Inf)
        end
        idx1 && (cache.gn_valid = true; cache.gn_norm = gn_norm)
        (Jᵀfu, gn, gn_norm)
    end

    if gn_norm <= Δ
        δu = Utils.restructure(δu, gn_step)
        set_du!(cache, δu, idx)
        extras = _more_extras(cache, J_, δu, zero(T))
        return DescentResult(; δu, extras)
    end

    # Moré's safeguarded Newton iteration on the damping parameter (MINPACK `lmpar`):
    # λ* ∈ (0, u₀] with u₀ = ‖D⁻¹ Jᵀfu‖ / Δ, since ‖D δu(λ)‖ ≤ ‖D⁻¹ Jᵀfu‖ / λ
    dtd = cache.dtd
    if dtd === nothing
        u_bound = cache.internalnorm(Jᵀfu) / Δ
    else
        if dtd isa Number
            u_bound = abs(Jᵀfu) / (sqrt(dtd) * Δ)
        else
            @bb @. cache.q = Jᵀfu / sqrt(dtd)
            u_bound = cache.internalnorm(cache.q) / Δ
        end
    end
    λ = if iszero(cache.λ)
        T(1.0e-3) * u_bound
    else
        min(T(cache.λ), u_bound)
    end
    # Below ~eps·maxdiag(JᵀJ)/min(diag DᵀD) the normal-equations factorization cannot
    # succeed; the augmented system stays full rank but still clamps λ off 0 to keep
    # the `Dp/√λ` right-hand side of the q-solve finite
    λ = if normal_form(cache)
        max(λ, eps(T) * _more_maxdiag(cache.JᵀJ) / _more_mindtd(dtd))
    else
        max(λ, eps(T))
    end
    l, uλ = zero(λ), max(u_bound, λ)
    λ_of_p = λ
    got_step = false

    @static_timeit cache.timer "more iteration" begin
        for i in 1:cache.maxiters
            linres = _more_damped_solve(cache, J_, Jᵀfu, fu, λ, u, kwargs)
            if !linres.success || !_all_finite(linres.u)
                # λ is numerically too small to regularize the system; the analytic
                # bound u₀ assumes exact arithmetic, so λ is allowed to outgrow it
                l = max(l, λ)
                λ *= 10
                uλ = max(uλ, λ)
                continue
            end
            p = linres.u
            if normal_form(cache)
                if p isa Number
                    cache.p = -p
                else
                    @bb @. cache.p = -p
                end
            else
                cache.p = Utils.restructure(cache.p, p)
            end
            p = cache.p
            got_step = true
            λ_of_p = λ

            Dp = _more_Dp!(cache, dtd, p)
            ϕ = cache.internalnorm(Dp) - Δ
            (abs(ϕ) <= cache.θ * Δ || i == cache.maxiters) && break

            D²p = _more_D2p!(cache, dtd, p)
            qres = _more_q_solve(cache, D²p, Dp, λ, u, kwargs)
            if !qres.success || !_all_finite(qres.u)
                l = max(l, λ)
                λ *= 10
                uλ = max(uλ, λ)
                continue
            end
            λ, l, uλ = _more_update_λ(
                λ, ϕ, Δ, Utils.safe_dot(Dp, Dp), Utils.safe_dot(D²p, qres.u), l, uλ
            )
        end
    end
    cache.λ = λ_of_p

    if !got_step
        set_du!(cache, δu, idx)
        return DescentResult(; δu, success = false, linsolve_success = false)
    end

    δu = Utils.restructure(δu, cache.p)
    set_du!(cache, δu, idx)
    extras = _more_extras(cache, J_, δu, λ_of_p)
    return DescentResult(; δu, extras)
end

# Undamped `min ‖Jp + fu‖` — the `λ = 0` augmented system. Normal form instead
# solves `JᵀJ p = Jᵀfu` on the same square cache (the step is negated after).
function _more_gn_solve(cache, J_, Jᵀfu, fu, gn_buf, u, kwargs)
    if cache.op_state !== nothing
        cache.op_state.λ = zero(cache.op_state.λ)
        b = normal_form(cache) ? Utils.safe_vec(Jᵀfu) :
            _more_augmented_rhs!(cache.rhs, fu)
        return cache.lincache(; b, linu = Utils.safe_vec(gn_buf), kwargs...)
    end
    if normal_form(cache)
        A0 = cache.JᵀJ isa AbstractMatrix ? Utils.maybe_symmetric(cache.JᵀJ) :
            cache.JᵀJ
        return cache.lincache(;
            A = A0, b = Utils.safe_vec(Jᵀfu), linu = Utils.safe_vec(gn_buf),
            reuse_A_if_factorization = false, kwargs...
        )
    end
    A = _more_augmented_system(
        J_, zero(promote_type(eltype(u), eltype(fu))), cache.dtd, cache.augmented, fu, u
    )
    b = _more_augmented_rhs!(cache.rhs, fu)
    return cache.lincache(;
        A, b, linu = Utils.safe_vec(gn_buf),
        reuse_A_if_factorization = false, kwargs...
    )
end

# Damped solve: normal form `(JᵀJ + λD²) p = Jᵀfu` (negated after); augmented
# `min ‖[J; √λD] p - [-fu; 0]‖`
function _more_damped_solve(cache, J_, Jᵀfu, fu, λ, u, kwargs)
    if cache.op_state !== nothing
        cache.op_state.λ = λ
        b = normal_form(cache) ? Utils.safe_vec(Jᵀfu) :
            _more_augmented_rhs!(cache.rhs, fu)
        return cache.lincache(; b, linu = Utils.safe_vec(cache.p), kwargs...)
    end
    if normal_form(cache)
        A = _more_damped_system(cache.JᵀJ, λ, cache.dtd, cache.damped, u)
        A = A isa AbstractMatrix ? Utils.maybe_symmetric(A) : A
        return cache.lincache(;
            A, b = Utils.safe_vec(Jᵀfu), linu = Utils.safe_vec(cache.p),
            reuse_A_if_factorization = false, kwargs...
        )
    end
    A = _more_augmented_system(J_, λ, cache.dtd, cache.augmented, fu, u)
    b = _more_augmented_rhs!(cache.rhs, fu)
    return cache.lincache(;
        A, b, linu = Utils.safe_vec(cache.p),
        reuse_A_if_factorization = false, kwargs...
    )
end

# `q = (JᵀJ + λD²)⁻¹ D²p`, so `pᵀD²q` feeds the Newton update. Normal form solves
# the square system directly; the augmented path uses `A \ [0; Dp/√λ]`, since
# `Aᵀ [0; Dp/√λ] = D²p` for `A = [J; √λD]`
function _more_q_solve(cache, D²p, Dp, λ, u, kwargs)
    if cache.op_state !== nothing || normal_form(cache)
        b = normal_form(cache) ? Utils.safe_vec(D²p) :
            _more_augmented_qrhs!(cache.qrhs, Dp, λ, length(cache.rhs) - length(cache.p))
        return cache.lincache(;
            b, linu = Utils.safe_vec(cache.q),
            reuse_A_if_factorization = true, kwargs...
        )
    end
    b = _more_augmented_qrhs!(cache.qrhs, Dp, λ, length(cache.rhs) - length(cache.p))
    return cache.lincache(;
        b, linu = Utils.safe_vec(cache.q),
        reuse_A_if_factorization = true, kwargs...
    )
end

# `extras` carries `λ` and the scaled step norm for `RadiusUpdateSchemes.More`, and the
# MINPACK-form predicted reduction `½‖Jδu‖² + λ‖Dδu‖²`, which avoids the cancellation
# of `-gᵀδu - ½δuᵀJᵀJδu` at a subproblem solution for ill-conditioned `J`
function _more_extras(cache, J_, δu, λ)
    if J_ isa Number
        δuJᵀJδu = abs2(J_ * δu)
    elseif cache.Jδu isa AbstractVector
        @bb cache.Jδu = J_ × Utils.safe_vec(δu)
        δuJᵀJδu = Utils.safe_dot(cache.Jδu, cache.Jδu)
    else
        δuJᵀJδu = Utils.safe_dot(J_ * Utils.safe_vec(δu), J_ * Utils.safe_vec(δu))
    end
    Dp = _more_Dp!(cache, cache.dtd, δu)
    predicted_reduction = δuJᵀJδu / 2 + λ * Utils.safe_dot(Dp, Dp)
    return (; λ, δuJᵀJδu, predicted_reduction, step_norm = cache.internalnorm(Dp))
end
