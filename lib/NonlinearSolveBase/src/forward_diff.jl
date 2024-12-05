@concrete mutable struct NonlinearSolveForwardDiffCache <: AbstractNonlinearSolveCache
    cache
    prob
    alg
    p
    values_p
    partials_p
end

function NonlinearSolveBase.get_abstol(cache::NonlinearSolveForwardDiffCache)
    NonlinearSolveBase.get_abstol(cache.cache)
end
function NonlinearSolveBase.get_reltol(cache::NonlinearSolveForwardDiffCache)
    NonlinearSolveBase.get_reltol(cache.cache)
end
