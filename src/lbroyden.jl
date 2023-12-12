"""
    LimitedMemoryBroyden(; max_resets::Int = 3, linesearch = nothing,
        threshold::Int = 10, reset_tolerance = nothing)

An implementation of `LimitedMemoryBroyden` with resetting and line search.

## Arguments

  - `max_resets`: the maximum number of resets to perform. Defaults to `3`.
  - `reset_tolerance`: the tolerance for the reset check. Defaults to
    `sqrt(eps(real(eltype(u))))`.
  - `threshold`: the number of vectors to store in the low rank approximation. Defaults
    to `10`.
  - `linesearch`: the line search algorithm to use. Defaults to [`LineSearch()`](@ref),
    which means that no line search is performed. Algorithms from `LineSearches.jl` can be
    used here directly, and they will be converted to the correct `LineSearch`. It is
    recommended to use [`LiFukushimaLineSearch`](@ref) -- a derivative free linesearch
    specifically designed for Broyden's method.
"""
@concrete struct LimitedMemoryBroyden{threshold} <: AbstractNewtonAlgorithm{false, Nothing}
    max_resets::Int
    linesearch
    reset_tolerance
end

function LimitedMemoryBroyden(; max_resets::Int = 3, linesearch = nothing,
        threshold::Union{Val, Int} = Val(27), reset_tolerance = nothing)
    linesearch = linesearch isa LineSearch ? linesearch : LineSearch(; method = linesearch)
    return LimitedMemoryBroyden{SciMLBase._unwrap_val(threshold)}(max_resets, linesearch,
        reset_tolerance)
end

__get_threshold(::LimitedMemoryBroyden{threshold}) where {threshold} = Val(threshold)
__get_unwrapped_threshold(::LimitedMemoryBroyden{threshold}) where {threshold} = threshold

@concrete mutable struct LimitedMemoryBroydenCache{iip} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    u_cache
    du
    fu
    fu_cache
    dfu
    p
    U
    Vᵀ
    threshold_cache
    mat_cache
    vᵀ_cache
    force_stop::Bool
    resets::Int
    iterations_since_reset::Int
    max_resets::Int
    maxiters::Int
    internalnorm
    retcode::ReturnCode.T
    abstol
    reltol
    reset_tolerance
    reset_check
    prob
    stats::NLStats
    ls_cache
    tc_cache
    trace
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::LimitedMemoryBroyden,
        args...; alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
        termination_condition = nothing, internalnorm::F = DEFAULT_NORM,
        kwargs...) where {uType, iip, F}
    @unpack f, u0, p = prob
    threshold = __get_threshold(alg)
    η = min(__get_unwrapped_threshold(alg), maxiters)
    if u0 isa Number || length(u0) ≤ η
        # If u is a number or very small problem then we simply use Broyden
        return SciMLBase.__init(prob,
            Broyden(; alg.max_resets, alg.reset_tolerance, alg.linesearch), args...;
            alias_u0, maxiters, abstol, internalnorm, kwargs...)
    end
    u = __maybe_unaliased(u0, alias_u0)
    fu = evaluate_f(prob, u)
    U, Vᵀ = __init_low_rank_jacobian(u, fu, threshold)

    @bb du = copy(fu)
    @bb u_cache = copy(u)
    @bb fu_cache = copy(fu)
    @bb dfu = similar(fu)
    @bb vᵀ_cache = similar(u)
    @bb mat_cache = similar(u)

    reset_tolerance = alg.reset_tolerance === nothing ? sqrt(eps(real(eltype(u)))) :
                      alg.reset_tolerance
    reset_check = x -> abs(x) ≤ reset_tolerance

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fu, u,
        termination_condition)

    U_part = selectdim(U, 1, 1:0)
    Vᵀ_part = selectdim(Vᵀ, 2, 1:0)
    trace = init_nonlinearsolve_trace(alg, u, fu, ApplyArray(*, Vᵀ_part, U_part), du;
        kwargs...)

    threshold_cache = __lbroyden_threshold_cache(u, threshold)

    return LimitedMemoryBroydenCache{iip}(f, alg, u, u_cache, du, fu, fu_cache, dfu, p,
        U, Vᵀ, threshold_cache, mat_cache, vᵀ_cache, false, 0, 0, alg.max_resets, maxiters,
        internalnorm, ReturnCode.Default, abstol, reltol, reset_tolerance, reset_check,
        prob, NLStats(1, 0, 0, 0, 0),
        init_linesearch_cache(alg.linesearch, f, u, p, fu, Val(iip)), tc_cache, trace)
end

function perform_step!(cache::LimitedMemoryBroydenCache{iip}) where {iip}
    T = eltype(cache.u)

    α = perform_linesearch!(cache.ls_cache, cache.u, cache.du)
    @bb axpy!(-α, cache.du, cache.u)
    evaluate_f(cache, cache.u, cache.p)

    idx = min(cache.iterations_since_reset, size(cache.U, 2))
    U_part = selectdim(cache.U, 2, 1:idx)
    Vᵀ_part = selectdim(cache.Vᵀ, 1, 1:idx)
    update_trace!(cache.trace, cache.stats.nsteps + 1, get_u(cache), cache.fu,
        ApplyArray(*, Vᵀ_part, U_part), cache.du, α)

    check_and_update!(cache, cache.fu, cache.u, cache.u_cache)

    cache.force_stop && return nothing

    # Update the Inverse Jacobian Approximation
    @bb @. cache.dfu = cache.fu - cache.fu_cache

    # Only try to reset if we have enough iterations since last reset
    if cache.iterations_since_reset > size(cache.U, 1) &&
       (all(cache.reset_check, cache.du) || all(cache.reset_check, cache.dfu))
        if cache.resets ≥ cache.max_resets
            cache.retcode = ReturnCode.ConvergenceFailure
            cache.force_stop = true
            return nothing
        end
        cache.iterations_since_reset = 0
        cache.resets += 1
        @bb copyto!(cache.du, cache.fu)
    else
        @bb cache.du .*= -1

        cache.vᵀ_cache = _rmatvec!!(cache.vᵀ_cache, cache.threshold_cache, U_part, Vᵀ_part,
            cache.du)
        cache.mat_cache = _matvec!!(cache.mat_cache, cache.threshold_cache, U_part, Vᵀ_part,
            cache.dfu)

        denom = dot(cache.vᵀ_cache, cache.dfu)
        @bb @. cache.u_cache = (cache.du - cache.mat_cache) /
                               ifelse(iszero(denom), T(1e-5), denom)

        idx = mod1(cache.iterations_since_reset + 1, size(cache.U, 2))
        selectdim(cache.U, 2, idx) .= _vec(cache.u_cache)
        selectdim(cache.Vᵀ, 1, idx) .= _vec(cache.vᵀ_cache)

        idx = min(cache.iterations_since_reset + 1, size(cache.U, 2))
        U_part = selectdim(cache.U, 2, 1:idx)
        Vᵀ_part = selectdim(cache.Vᵀ, 1, 1:idx)
        cache.du = _matvec!!(cache.du, cache.threshold_cache, U_part, Vᵀ_part, cache.fu)

        cache.iterations_since_reset += 1
    end

    @bb copyto!(cache.u_cache, cache.u)
    @bb copyto!(cache.fu_cache, cache.fu)

    return nothing
end

function __reinit_internal!(cache::LimitedMemoryBroydenCache; kwargs...)
    cache.iterations_since_reset = 0
    return nothing
end

function _rmatvec!!(y, xᵀU, U, Vᵀ, x)
    # xᵀ × (-I + UVᵀ)
    η = size(U, 2)
    if η == 0
        @bb @. y = -x
        return y
    end
    x_ = vec(x)
    xᵀU_ = view(xᵀU, 1:η)
    @bb xᵀU_ = transpose(U) × x_
    @bb y = transpose(Vᵀ) × vec(xᵀU_)
    @bb @. y -= x
    return y
end

function _matvec!!(y, Vᵀx, U, Vᵀ, x)
    # (-I + UVᵀ) × x
    η = size(U, 2)
    if η == 0
        @bb @. y = -x
        return y
    end
    x_ = vec(x)
    Vᵀx_ = view(Vᵀx, 1:η)
    @bb Vᵀx_ = Vᵀ × x_
    @bb y = U × vec(Vᵀx_)
    @bb @. y -= x
    return y
end

@inline function __lbroyden_threshold_cache(x, ::Val{threshold}) where {threshold}
    return similar(x, threshold)
end
@inline function __lbroyden_threshold_cache(x::SArray, ::Val{threshold}) where {threshold}
    return zeros(SVector{threshold, eltype(x)})
end
