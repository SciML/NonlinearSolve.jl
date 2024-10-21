function callback_into_cache!(topcache, cache::AbstractLineSearchCache, args...)
    LineSearch.callback_into_cache!(cache, get_fu(topcache))
end

function reinit_cache!(cache::AbstractLineSearchCache, args...; kwargs...)
    return SciMLBase.reinit!(cache, args...; kwargs...)
end
