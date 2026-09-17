@concrete mutable struct NonlinearSolveForwardDiffCache <: AbstractNonlinearSolveCache
    cache
    prob
    alg
    p
    values_p
    # Construction-time partials of `p`; write-only bookkeeping. `reinit!`
    # refreshes it only when the new partials match this concrete type.
    partials_p
    # Dual leaves of the original `u0` (or `tspan`), when only those carried
    # duals — needed to recover the tag for the (structurally zero)
    # initial-guess partials of a converged root.
    u0duals::Union{Nothing, AbstractVector}
end

NonlinearSolveForwardDiffCache(cache, prob, alg, p, values_p, partials_p) =
    NonlinearSolveForwardDiffCache(cache, prob, alg, p, values_p, partials_p, nothing)

get_u(cache::NonlinearSolveForwardDiffCache) = get_u(cache.cache)
get_fu(cache::NonlinearSolveForwardDiffCache) = get_fu(cache.cache)
set_fu!(cache::NonlinearSolveForwardDiffCache, fu) = set_fu!(cache.cache, fu)
SciMLBase.set_u!(cache::NonlinearSolveForwardDiffCache, u) = SciMLBase.set_u!(cache.cache, u)
function _solve_without_solution!(cache::NonlinearSolveForwardDiffCache)
    return CommonSolve.solve!(cache)
end
function NonlinearSolveBase.get_abstol(cache::NonlinearSolveForwardDiffCache)
    return NonlinearSolveBase.get_abstol(cache.cache)
end
function NonlinearSolveBase.get_reltol(cache::NonlinearSolveForwardDiffCache)
    return NonlinearSolveBase.get_reltol(cache.cache)
end
