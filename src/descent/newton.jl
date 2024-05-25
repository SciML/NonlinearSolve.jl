"""
    NewtonDescent(; linsolve = nothing, precs = DEFAULT_PRECS)

Compute the descent direction as ``J δu = -fu``. For non-square Jacobian problems, this is
commonly referred to as the Gauss-Newton Descent.

See also [`Dogleg`](@ref), [`SteepestDescent`](@ref), [`DampedNewtonDescent`](@ref).
"""
@kwdef @concrete struct NewtonDescent <: AbstractDescentAlgorithm
    linsolve = nothing
    precs = DEFAULT_PRECS
end

function Base.show(io::IO, d::NewtonDescent)
    modifiers = String[]
    d.linsolve !== nothing && push!(modifiers, "linsolve = $(d.linsolve)")
    d.precs !== DEFAULT_PRECS && push!(modifiers, "precs = $(d.precs)")
    print(io, "NewtonDescent($(join(modifiers, ", ")))")
end

supports_line_search(::NewtonDescent) = true

@concrete mutable struct NewtonDescentCache{pre_inverted, normalform} <:
                         AbstractDescentCache
    δu
    δus
    lincache
    JᵀJ_cache  # For normal form else nothing
    Jᵀfu_cache
    timer
end

@internal_caches NewtonDescentCache :lincache

function __internal_init(prob::NonlinearProblem, alg::NewtonDescent, J, fu, u; stats,
        shared::Val{N} = Val(1), pre_inverted::Val{INV} = False,
        linsolve_kwargs = (;), abstol = nothing, reltol = nothing,
        timer = get_timer_output(), kwargs...) where {INV, N}
    @bb δu = similar(u)
    δus = N ≤ 1 ? nothing : map(2:N) do i
        @bb δu_ = similar(u)
    end
    INV && return NewtonDescentCache{true, false}(δu, δus, nothing, nothing, nothing, timer)
    lincache = LinearSolverCache(
        alg, alg.linsolve, J, _vec(fu), _vec(u); stats, abstol, reltol, linsolve_kwargs...)
    return NewtonDescentCache{false, false}(δu, δus, lincache, nothing, nothing, timer)
end

function __internal_init(prob::NonlinearLeastSquaresProblem, alg::NewtonDescent, J, fu,
        u; stats, pre_inverted::Val{INV} = False, linsolve_kwargs = (;),
        shared::Val{N} = Val(1), abstol = nothing, reltol = nothing,
        timer = get_timer_output(), kwargs...) where {INV, N}
    length(fu) != length(u) &&
        @assert !INV "Precomputed Inverse for Non-Square Jacobian doesn't make sense."

    normal_form = __needs_square_A(alg.linsolve, u)
    if normal_form
        JᵀJ = transpose(J) * J
        Jᵀfu = transpose(J) * _vec(fu)
        A, b = __maybe_symmetric(JᵀJ), Jᵀfu
    else
        JᵀJ, Jᵀfu = nothing, nothing
        A, b = J, _vec(fu)
    end
    lincache = LinearSolverCache(
        alg, alg.linsolve, A, b, _vec(u); stats, abstol, reltol, linsolve_kwargs...)
    @bb δu = similar(u)
    δus = N ≤ 1 ? nothing : map(2:N) do i
        @bb δu_ = similar(u)
    end
    return NewtonDescentCache{false, normal_form}(δu, δus, lincache, JᵀJ, Jᵀfu, timer)
end

function __internal_solve!(
        cache::NewtonDescentCache{INV, false}, J, fu, u, idx::Val = Val(1);
        skip_solve::Bool = false, new_jacobian::Bool = true, kwargs...) where {INV}
    δu = get_du(cache, idx)
    skip_solve && return DescentResult(; δu)
    if INV
        @assert J!==nothing "`J` must be provided when `pre_inverted = Val(true)`."
        @bb δu = J × vec(fu)
    else
        @static_timeit cache.timer "linear solve" begin
            linres = cache.lincache(;
                A = J, b = _vec(fu), kwargs..., linu = _vec(δu), du = _vec(δu),
                reuse_A_if_factorization = !new_jacobian || (idx !== Val(1)))
            δu = _restructure(get_du(cache, idx), linres.u)
            if !linres.success
                set_du!(cache, δu, idx)
                return DescentResult(; δu, success = false, linsolve_success = false)
            end
        end
    end
    @bb @. δu *= -1
    set_du!(cache, δu, idx)
    return DescentResult(; δu)
end

function __internal_solve!(
        cache::NewtonDescentCache{false, true}, J, fu, u, idx::Val = Val(1);
        skip_solve::Bool = false, new_jacobian::Bool = true, kwargs...)
    δu = get_du(cache, idx)
    skip_solve && return DescentResult(; δu)
    if idx === Val(1)
        @bb cache.JᵀJ_cache = transpose(J) × J
    end
    @bb cache.Jᵀfu_cache = transpose(J) × vec(fu)
    @static_timeit cache.timer "linear solve" begin
        linres = cache.lincache(;
            A = __maybe_symmetric(cache.JᵀJ_cache), b = cache.Jᵀfu_cache,
            kwargs..., linu = _vec(δu), du = _vec(δu),
            reuse_A_if_factorization = !new_jacobian || (idx !== Val(1)))
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
