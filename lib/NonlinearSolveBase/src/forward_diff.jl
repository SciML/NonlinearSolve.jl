@concrete mutable struct NonlinearSolveForwardDiffCache <: AbstractNonlinearSolveCache
    cache
    prob
    alg
    p
    values_p
    partials_p
end

get_u(cache::NonlinearSolveForwardDiffCache) = get_u(cache.cache)
get_fu(cache::NonlinearSolveForwardDiffCache) = get_fu(cache.cache)
set_fu!(cache::NonlinearSolveForwardDiffCache, fu) = set_fu!(cache.cache, fu)
SciMLBase.set_u!(cache::NonlinearSolveForwardDiffCache, u) = SciMLBase.set_u!(cache.cache, u)
function NonlinearSolveBase.get_abstol(cache::NonlinearSolveForwardDiffCache)
    return NonlinearSolveBase.get_abstol(cache.cache)
end
function NonlinearSolveBase.get_reltol(cache::NonlinearSolveForwardDiffCache)
    return NonlinearSolveBase.get_reltol(cache.cache)
end
