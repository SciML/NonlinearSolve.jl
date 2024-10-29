module NonlinearSolveBaseLineSearchExt

using LineSearch: LineSearch, AbstractLineSearchCache
using SciMLBase: SciMLBase

using NonlinearSolveBase: NonlinearSolveBase, InternalAPI

function NonlinearSolveBase.callback_into_cache!(
        topcache, cache::AbstractLineSearchCache, args...
)
    return LineSearch.callback_into_cache!(cache, NonlinearSolveBase.get_fu(topcache))
end

function InternalAPI.reinit!(cache::AbstractLineSearchCache; kwargs...)
    return SciMLBase.reinit!(cache; kwargs...)
end

end
