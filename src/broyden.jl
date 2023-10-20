# Sadly `Broyden` is taken up by SimpleNonlinearSolve.jl
"""
    GeneralBroyden(max_resets)
    GeneralBroyden(; max_resets = 3)

An implementation of `Broyden` with support for caching!

## Arguments

  - `max_resets`: the maximum number of resets to perform. Defaults to `3`.
"""
struct GeneralBroyden <: AbstractNewtonAlgorithm{false, Nothing}
    max_resets::Int
end

GeneralBroyden(; max_resets = 3) = GeneralBroyden(max_resets)

@concrete mutable struct GeneralBroydenCache{iip} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    du
    fu
    fu2
    dfu
    p
    J⁻¹
    J⁻¹₂
    J⁻¹df
    force_stop::Bool
    resets::Int
    max_rests::Int
    maxiters::Int
    internalnorm
    retcode::ReturnCode.T
    abstol
    prob
    stats::NLStats
end

get_fu(cache::GeneralBroydenCache) = cache.fu

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::GeneralBroyden, args...;
    alias_u0 = false, maxiters = 1000, abstol = 1e-6, internalnorm = DEFAULT_NORM,
    kwargs...) where {uType, iip}
    @unpack f, u0, p = prob
    u = alias_u0 ? u0 : deepcopy(u0)
    fu = evaluate_f(prob, u)
    J⁻¹ = convert(parameterless_type(_mutable(u)),
        Matrix{eltype(u)}(I, length(fu), length(u)))
    return GeneralBroydenCache{iip}(f, alg, u, _mutable_zero(u), fu, similar(fu),
        similar(fu), p, J⁻¹, similar(fu'), _mutable_zero(u), false, 0, alg.max_resets,
        maxiters, internalnorm, ReturnCode.Default, abstol, prob, NLStats(1, 0, 0, 0, 0))
end

function perform_step!(cache::GeneralBroydenCache{true})
    @unpack f, p, du, fu, fu2, dfu, u, J⁻¹, J⁻¹df, J⁻¹₂ = cache
    T = eltype(u)

    mul!(du, J⁻¹, -fu)
    u .+= du
    f(fu2, u, p)

    cache.internalnorm(fu2) < cache.abstol && (cache.force_stop = true)
    cache.stats.nf += 1

    cache.force_stop && return nothing

    # Update the inverse jacobian
    dfu .= fu2 .- fu
    if cache.resets < cache.max_rests &&
       (all(x -> abs(x) ≤ 1e-12, du) || all(x -> abs(x) ≤ 1e-12, dfu))
        fill!(J⁻¹, 0)
        J⁻¹[diagind(J⁻¹)] .= T(1)
        cache.resets += 1
    else
        mul!(J⁻¹df, J⁻¹, dfu)
        mul!(J⁻¹₂, du', J⁻¹)
        du .= (du .- J⁻¹df) ./ (dot(du, J⁻¹df) .+ T(1e-5))
        mul!(J⁻¹, reshape(du, :, 1), J⁻¹₂, 1, 1)
    end
    fu .= fu2

    return nothing
end
