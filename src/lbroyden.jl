"""
    LimitedMemoryBroyden(; max_resets::Int = 3, linesearch = LineSearch(),
        threshold::Int = 10, reset_tolerance = nothing)

An implementation of `LimitedMemoryBroyden` with reseting and line search.

## Arguments

  - `max_resets`: the maximum number of resets to perform. Defaults to `3`.
  - `reset_tolerance`: the tolerance for the reset check. Defaults to
    `sqrt(eps(eltype(u)))`.
  - `threshold`: the number of vectors to store in the low rank approximation. Defaults
    to `10`.
  - `linesearch`: the line search algorithm to use. Defaults to [`LineSearch()`](@ref),
    which means that no line search is performed. Algorithms from `LineSearches.jl` can be
    used here directly, and they will be converted to the correct `LineSearch`. It is
    recommended to use [LiFukushimaLineSearchCache](@ref) -- a derivative free linesearch
    specifically designed for Broyden's method.
"""
@concrete struct LimitedMemoryBroyden <: AbstractNewtonAlgorithm{false, Nothing}
    max_resets::Int
    threshold::Int
    linesearch
    reset_tolerance
end

function LimitedMemoryBroyden(; max_resets::Int = 3, linesearch = LineSearch(),
    threshold::Int = 10, reset_tolerance = nothing)
    linesearch = linesearch isa LineSearch ? linesearch : LineSearch(; method = linesearch)
    return LimitedMemoryBroyden(max_resets, threshold, linesearch, reset_tolerance)
end

@concrete mutable struct LimitedMemoryBroydenCache{iip} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    du
    fu
    fu2
    dfu
    p
    U
    Vᵀ
    Ux
    xᵀVᵀ
    u_cache
    vᵀ_cache
    force_stop::Bool
    resets::Int
    iterations_since_reset::Int
    max_resets::Int
    maxiters::Int
    internalnorm
    retcode::ReturnCode.T
    abstol
    reset_tolerance
    reset_check
    prob
    stats::NLStats
    lscache
end

get_fu(cache::LimitedMemoryBroydenCache) = cache.fu

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::LimitedMemoryBroyden,
    args...; alias_u0 = false, maxiters = 1000, abstol = 1e-6, internalnorm = DEFAULT_NORM,
    kwargs...) where {uType, iip}
    @unpack f, u0, p = prob
    u = alias_u0 ? u0 : deepcopy(u0)
    if u isa Number
        # If u is a number then we simply use Broyden
        return SciMLBase.__init(prob,
            GeneralBroyden(; alg.max_resets, alg.reset_tolerance,
                alg.linesearch), args...; alias_u0, maxiters, abstol, internalnorm, kwargs...)
    end
    fu = evaluate_f(prob, u)
    threshold = min(alg.threshold, maxiters)
    U, Vᵀ = __init_low_rank_jacobian(u, fu, threshold)
    du = -fu
    reset_tolerance = alg.reset_tolerance === nothing ? sqrt(eps(eltype(u))) :
                      alg.reset_tolerance
    reset_check = x -> abs(x) ≤ reset_tolerance
    return LimitedMemoryBroydenCache{iip}(f, alg, u, du, fu, zero(fu),
        zero(fu), p, U, Vᵀ, similar(u, threshold), similar(u, 1, threshold),
        zero(u), zero(u), false, 0, 0, alg.max_resets, maxiters, internalnorm,
        ReturnCode.Default, abstol, reset_tolerance, reset_check, prob,
        NLStats(1, 0, 0, 0, 0),
        init_linesearch_cache(alg.linesearch, f, u, p, fu, Val(iip)))
end

function perform_step!(cache::LimitedMemoryBroydenCache{true})
    @unpack f, p, du, u = cache
    T = eltype(u)

    α = perform_linesearch!(cache.lscache, u, du)
    axpy!(α, du, u)
    f(cache.fu2, u, p)

    cache.internalnorm(cache.fu2) < cache.abstol && (cache.force_stop = true)
    cache.stats.nf += 1

    cache.force_stop && return nothing

    # Update the Inverse Jacobian Approximation
    cache.dfu .= cache.fu2 .- cache.fu

    # Only try to reset if we have enough iterations since last reset
    if cache.iterations_since_reset > size(cache.U, 1) &&
       (all(cache.reset_check, du) || all(cache.reset_check, cache.dfu))
        if cache.resets ≥ cache.max_resets
            cache.retcode = ReturnCode.Unstable
            cache.force_stop = true
            return nothing
        end
        cache.iterations_since_reset = 0
        cache.resets += 1
        cache.du .= -cache.fu
    else
        idx = min(cache.iterations_since_reset, size(cache.U, 1))
        U_part = selectdim(cache.U, 1, 1:idx)
        Vᵀ_part = selectdim(cache.Vᵀ, 2, 1:idx)

        __lbroyden_matvec!(_vec(cache.vᵀ_cache), cache.Ux, U_part, Vᵀ_part, _vec(cache.du))
        __lbroyden_rmatvec!(_vec(cache.u_cache), cache.xᵀVᵀ, U_part, Vᵀ_part,
            _vec(cache.dfu))
        cache.u_cache .= (du .- cache.u_cache) ./
                         (dot(cache.vᵀ_cache, cache.dfu) .+ T(1e-5))

        idx = mod1(cache.iterations_since_reset + 1, size(cache.U, 1))
        selectdim(cache.U, 1, idx) .= _vec(cache.u_cache)
        selectdim(cache.Vᵀ, 2, idx) .= _vec(cache.vᵀ_cache)

        idx = min(cache.iterations_since_reset + 1, size(cache.U, 1))
        U_part = selectdim(cache.U, 1, 1:idx)
        Vᵀ_part = selectdim(cache.Vᵀ, 2, 1:idx)
        __lbroyden_matvec!(_vec(cache.du), cache.Ux, U_part, Vᵀ_part, _vec(cache.fu2))
        cache.du .*= -1
        cache.iterations_since_reset += 1
    end

    cache.fu .= cache.fu2

    return nothing
end

function perform_step!(cache::LimitedMemoryBroydenCache{false})
    @unpack f, p = cache
    T = eltype(cache.u)

    α = perform_linesearch!(cache.lscache, cache.u, cache.du)
    cache.u = cache.u .+ α * cache.du
    cache.fu2 = f(cache.u, p)

    cache.internalnorm(cache.fu2) < cache.abstol && (cache.force_stop = true)
    cache.stats.nf += 1

    cache.force_stop && return nothing

    # Update the Inverse Jacobian Approximation
    cache.dfu .= cache.fu2 .- cache.fu

    # Only try to reset if we have enough iterations since last reset
    if cache.iterations_since_reset > size(cache.U, 1) &&
       (all(cache.reset_check, cache.du) || all(cache.reset_check, cache.dfu))
        if cache.resets ≥ cache.max_resets
            cache.retcode = ReturnCode.Unstable
            cache.force_stop = true
            return nothing
        end
        cache.iterations_since_reset = 0
        cache.resets += 1
        cache.du = -cache.fu
    else
        idx = min(cache.iterations_since_reset, size(cache.U, 1))
        U_part = selectdim(cache.U, 1, 1:idx)
        Vᵀ_part = selectdim(cache.Vᵀ, 2, 1:idx)

        cache.vᵀ_cache = _restructure(cache.vᵀ_cache,
            __lbroyden_matvec(U_part, Vᵀ_part, _vec(cache.du)))
        cache.u_cache = _restructure(cache.u_cache,
            __lbroyden_rmatvec(U_part, Vᵀ_part, _vec(cache.dfu)))
        cache.u_cache = (cache.du .- cache.u_cache) ./
                        (dot(cache.vᵀ_cache, cache.dfu) .+ T(1e-5))

        idx = mod1(cache.iterations_since_reset + 1, size(cache.U, 1))
        selectdim(cache.U, 1, idx) .= _vec(cache.u_cache)
        selectdim(cache.Vᵀ, 2, idx) .= _vec(cache.vᵀ_cache)

        idx = min(cache.iterations_since_reset + 1, size(cache.U, 1))
        U_part = selectdim(cache.U, 1, 1:idx)
        Vᵀ_part = selectdim(cache.Vᵀ, 2, 1:idx)
        cache.du = _restructure(cache.du,
            -__lbroyden_matvec(U_part, Vᵀ_part, _vec(cache.fu2)))
        cache.iterations_since_reset += 1
    end

    cache.fu = cache.fu2

    return nothing
end

function SciMLBase.reinit!(cache::LimitedMemoryBroydenCache{iip}, u0 = cache.u; p = cache.p,
    abstol = cache.abstol, maxiters = cache.maxiters) where {iip}
    cache.p = p
    if iip
        recursivecopy!(cache.u, u0)
        cache.f(cache.fu, cache.u, p)
    else
        # don't have alias_u0 but cache.u is never mutated for OOP problems so it doesn't matter
        cache.u = u0
        cache.fu = cache.f(cache.u, p)
    end
    cache.abstol = abstol
    cache.maxiters = maxiters
    cache.stats.nf = 1
    cache.stats.nsteps = 1
    cache.resets = 0
    cache.iterations_since_reset = 0
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
    return cache
end

@views function __lbroyden_matvec!(y::AbstractVector, Ux::AbstractVector,
    U::AbstractMatrix, Vᵀ::AbstractMatrix, x::AbstractVector)
    # Computes Vᵀ × U × x
    η = size(U, 1)
    if η == 0
        y .= x
        return nothing
    end
    mul!(Ux[1:η], U, x)
    mul!(y, Vᵀ[:, 1:η], Ux[1:η])
    return nothing
end

@views function __lbroyden_matvec(U::AbstractMatrix, Vᵀ::AbstractMatrix, x::AbstractVector)
    # Computes Vᵀ × U × x
    size(U, 1) == 0 && return x
    return Vᵀ * (U * x)
end

@views function __lbroyden_rmatvec!(y::AbstractVector, xᵀVᵀ::AbstractMatrix,
    U::AbstractMatrix, Vᵀ::AbstractMatrix, x::AbstractVector)
    # Computes xᵀ × Vᵀ × U
    η = size(U, 1)
    if η == 0
        y .= x
        return nothing
    end
    mul!(xᵀVᵀ[:, 1:η], x', Vᵀ)
    mul!(reshape(y, 1, :), xᵀVᵀ[:, 1:η], U)
    return nothing
end

@views function __lbroyden_rmatvec(U::AbstractMatrix, Vᵀ::AbstractMatrix, x::AbstractVector)
    # Computes xᵀ × Vᵀ × U
    size(U, 1) == 0 && return x
    return (reshape(x, 1, :) * Vᵀ) * U
end
