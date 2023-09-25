@concrete struct PseudoTransient{CJ, AD} <: AbstractNewtonAlgorithm{CJ, AD}
    ad::AD
    linsolve
    precs
    alpha_initial
end

concrete_jac(::PseudoTransient{CJ}) where {CJ} = CJ

function PseudoTransient(; concrete_jac = nothing, linsolve = nothing,
    precs = DEFAULT_PRECS, alpha_initial = 1e-3, adkwargs...)
    ad = default_adargs_to_adtype(; adkwargs...)
    return PseudoTransient{_unwrap_val(concrete_jac)}(ad, linsolve, precs, alpha_initial)
end

@concrete mutable struct PseudoTransientCache{iip}
    f
    alg
    u
    fu1
    fu2
    du
    p
    alpha
    res_norm
    uf
    linsolve
    J
    jac_cache
    force_stop
    maxiters::Int
    internalnorm
    retcode::ReturnCode.T
    abstol
    prob
    stats::NLStats
end

isinplace(::PseudoTransientCache{iip}) where {iip} = iip

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::PseudoTransient, args...;
    alias_u0 = false, maxiters = 1000, abstol = 1e-6, internalnorm = DEFAULT_NORM,
    kwargs...) where {uType, iip}
    @unpack f, u0, p = prob
    u = alias_u0 ? u0 : deepcopy(u0)
    if iip
        fu1 = f.resid_prototype === nothing ? zero(u) : f.resid_prototype
        f(fu1, u, p)
    else
        fu1 = _mutable(f(u, p))
    end
    uf, linsolve, J, fu2, jac_cache, du = jacobian_caches(alg, f, u, p, Val(iip))
    alpha = convert(eltype(u), alg.alpha_initial)
    res_norm = internalnorm(fu1)

    return PseudoTransientCache{iip}(f, alg, u, fu1, fu2, du, p, alpha, res_norm, uf,
        linsolve, J,
        jac_cache, false, maxiters, internalnorm, ReturnCode.Default, abstol, prob,
        NLStats(1, 0, 0, 0, 0))
end

function perform_step!(cache::PseudoTransientCache{true})
    @unpack u, fu1, f, p, alg, J, linsolve, du, alpha = cache
    jacobian!!(J, cache)
    J_new = J - (1 / alpha) * I

    # u = u - J \ fu
    linres = dolinsolve(alg.precs, linsolve; A = J_new, b = _vec(fu1), linu = _vec(du),
        p, reltol = cache.abstol)
    cache.linsolve = linres.cache
    @. u = u - du
    f(fu1, u, p)

    new_norm = cache.internalnorm(fu1)
    cache.alpha *= cache.res_norm / new_norm
    cache.res_norm = new_norm

    new_norm < cache.abstol && (cache.force_stop = true)
    cache.stats.nf += 1
    cache.stats.njacs += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

function perform_step!(cache::PseudoTransientCache{false})
    @unpack u, fu1, f, p, alg, linsolve, alpha = cache

    cache.J = jacobian!!(cache.J, cache)
    # u = u - J \ fu
    if linsolve === nothing
        cache.du = fu1 / (cache.J - (1 / alpha) * I)
    else
        linres = dolinsolve(alg.precs, linsolve; A = cache.J - (1 / alpha) * I,
            b = _vec(fu1),
            linu = _vec(cache.du), p, reltol = cache.abstol)
        cache.linsolve = linres.cache
    end
    cache.u = @. u - cache.du  # `u` might not support mutation
    cache.fu1 = f(cache.u, p)

    new_norm = cache.internalnorm(fu1)
    cache.alpha *= cache.res_norm / new_norm
    cache.res_norm = new_norm

    new_norm < cache.abstol && (cache.force_stop = true)
    cache.stats.nf += 1
    cache.stats.njacs += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

function SciMLBase.solve!(cache::PseudoTransientCache)
    while !cache.force_stop && cache.stats.nsteps < cache.maxiters
        perform_step!(cache)
        cache.stats.nsteps += 1
    end

    if cache.stats.nsteps == cache.maxiters
        cache.retcode = ReturnCode.MaxIters
    else
        cache.retcode = ReturnCode.Success
    end

    return SciMLBase.build_solution(cache.prob, cache.alg, cache.u, cache.fu1;
        cache.retcode, cache.stats)
end

function SciMLBase.reinit!(cache::PseudoTransientCache{iip}, u0 = cache.u; p = cache.p,
    abstol = cache.abstol, maxiters = cache.maxiters) where {iip}
    cache.p = p
    if iip
        recursivecopy!(cache.u, u0)
        cache.f(cache.fu1, cache.u, p)
    else
        # don't have alias_u0 but cache.u is never mutated for OOP problems so it doesn't matter
        cache.u = u0
        cache.fu1 = cache.f(cache.u, p)
    end
    cache.alpha = convert(eltype(u), 1e-3)
    cache.res_norm = internalnorm(cache.fu1)
    cache.abstol = abstol
    cache.maxiters = maxiters
    cache.stats.nf = 1
    cache.stats.nsteps = 1
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
    return cache
end
