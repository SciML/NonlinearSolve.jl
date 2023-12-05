# Sadly `Broyden` is taken up by SimpleNonlinearSolve.jl
"""
    GeneralBroyden(; max_resets = 3, linesearch = nothing, reset_tolerance = nothing,
        init_jacobian::Val = Val(:identity), autodiff = nothing)

An implementation of `Broyden` with resetting and line search.

## Arguments

  - `max_resets`: the maximum number of resets to perform. Defaults to `3`.
  - `reset_tolerance`: the tolerance for the reset check. Defaults to
    `sqrt(eps(real(eltype(u))))`.
  - `linesearch`: the line search algorithm to use. Defaults to [`LineSearch()`](@ref),
    which means that no line search is performed. Algorithms from `LineSearches.jl` can be
    used here directly, and they will be converted to the correct `LineSearch`. It is
    recommended to use [LiFukushimaLineSearch](@ref) -- a derivative free linesearch
    specifically designed for Broyden's method.
  - `init_jacobian`: the method to use for initializing the jacobian. Defaults to using the
    identity matrix (`Val(:identitiy)`). Alternatively, can be set to `Val(:true_jacobian)`
    to use the true jacobian as initialization.
  - `autodiff`: determines the backend used for the Jacobian. Note that this argument is
    ignored if an analytical Jacobian is passed, as that will be used instead. Defaults to
    `nothing` which means that a default is selected according to the problem specification!
    Valid choices are types from ADTypes.jl. (Used if `init_jacobian = Val(:true_jacobian)`)
"""
@concrete struct GeneralBroyden{IJ, CJ, AD} <: AbstractNewtonAlgorithm{CJ, AD}
    ad::AD
    max_resets::Int
    reset_tolerance
    linesearch
end

function set_ad(alg::GeneralBroyden{IJ, CJ}, ad) where {IJ, CJ}
    return GeneralBroyden{IJ, CJ}(ad, alg.max_resets, alg.reset_tolerance, alg.linesearch)
end

function GeneralBroyden(; max_resets = 3, linesearch = nothing,
        reset_tolerance = nothing, init_jacobian::Val = Val(:identity),
        autodiff = nothing)
    IJ = _unwrap_val(init_jacobian)
    @assert IJ ∈ (:identity, :true_jacobian)
    linesearch = linesearch isa LineSearch ? linesearch : LineSearch(; method = linesearch)
    CJ = IJ === :true_jacobian
    return GeneralBroyden{IJ, CJ}(autodiff, max_resets, reset_tolerance, linesearch)
end

@concrete mutable struct GeneralBroydenCache{iip, IJ} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    u_cache
    du
    fu
    fu_cache
    dfu
    p
    uf
    J⁻¹
    J⁻¹dfu
    force_stop::Bool
    resets::Int
    max_resets::Int
    maxiters::Int
    internalnorm
    retcode::ReturnCode.T
    abstol
    reltol
    reset_tolerance
    reset_check
    jac_cache
    prob
    stats::NLStats
    ls_cache
    tc_cache
    trace
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg_::GeneralBroyden{IJ},
        args...; alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
        termination_condition = nothing, internalnorm::F = DEFAULT_NORM,
        kwargs...) where {uType, iip, F, IJ}
    @unpack f, u0, p = prob
    u = __maybe_unaliased(u0, alias_u0)
    fu = evaluate_f(prob, u)
    @bb du = copy(u)

    if IJ === :true_jacobian
        alg = get_concrete_algorithm(alg_, prob)
        uf, _, J, fu_cache, jac_cache, du = jacobian_caches(alg, f, u, p, Val(iip);
            lininit = Val(false))
        J⁻¹ = J
    else
        alg = alg_
        @bb du = similar(u)
        uf, fu_cache, jac_cache = nothing, nothing, nothing
        J⁻¹ = __init_identity_jacobian(u, fu)
    end

    reset_tolerance = alg.reset_tolerance === nothing ? sqrt(eps(real(eltype(u)))) :
                      alg.reset_tolerance
    reset_check = x -> abs(x) ≤ reset_tolerance

    @bb u_cache = copy(u)
    @bb dfu = copy(fu)
    @bb J⁻¹dfu = similar(u)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fu, u,
        termination_condition)
    trace = init_nonlinearsolve_trace(alg, u, fu, J⁻¹, du; uses_jac_inverse = Val(true),
        kwargs...)

    return GeneralBroydenCache{iip, IJ}(f, alg, u, u_cache, du, fu, fu_cache, dfu, p, uf,
        J⁻¹, J⁻¹dfu, false, 0, alg.max_resets, maxiters, internalnorm, ReturnCode.Default,
        abstol, reltol, reset_tolerance, reset_check, jac_cache, prob,
        NLStats(1, 0, 0, 0, 0),
        init_linesearch_cache(alg.linesearch, f, u, p, fu, Val(iip)), tc_cache, trace)
end

function perform_step!(cache::GeneralBroydenCache{iip, IJ}) where {iip, IJ}
    T = eltype(cache.u)

    if IJ === :true_jacobian && cache.stats.nsteps == 0
        cache.J⁻¹ = inv(jacobian!!(cache.J⁻¹, cache)) # This allocates
    end

    @bb cache.du = cache.J⁻¹ × vec(cache.fu)
    α = perform_linesearch!(cache.ls_cache, cache.u, cache.du)
    @bb axpy!(-α, cache.du, cache.u)

    evaluate_f(cache, cache.u, cache.p)

    update_trace!(cache, α)
    check_and_update!(cache, cache.fu, cache.u, cache.u_cache)

    cache.force_stop && return nothing

    # Update the inverse jacobian
    @bb @. cache.dfu = cache.fu - cache.dfu

    if all(cache.reset_check, cache.du) || all(cache.reset_check, cache.dfu)
        if cache.resets ≥ cache.max_resets
            cache.retcode = ReturnCode.ConvergenceFailure
            cache.force_stop = true
            return nothing
        end
        if IJ === :true_jacobian
            cache.J⁻¹ = inv(jacobian!!(cache.J⁻¹, cache))
        else
            cache.J⁻¹ = __reinit_identity_jacobian!!(cache.J⁻¹)
        end
        cache.resets += 1
    else
        @bb cache.du .*= -1
        @bb cache.J⁻¹dfu = cache.J⁻¹ × vec(cache.dfu)
        @bb cache.u_cache = transpose(cache.J⁻¹) × vec(cache.du)
        denom = dot(cache.du, cache.J⁻¹dfu)
        @bb @. cache.du = (cache.du - cache.J⁻¹dfu) / ifelse(iszero(denom), T(1e-5), denom)
        @bb cache.J⁻¹ += vec(cache.du) × transpose(_vec(cache.u_cache))
    end

    @bb copyto!(cache.dfu, cache.fu)
    @bb copyto!(cache.u_cache, cache.u)

    return nothing
end

function __reinit_internal!(cache::GeneralBroydenCache; kwargs...)
    cache.J⁻¹ = __reinit_identity_jacobian!!(cache.J⁻¹)
    cache.resets = 0
    return nothing
end
