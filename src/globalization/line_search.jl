LineSearchesJL(method; kwargs...) = LineSearchesJL(; method, kwargs...)
function LineSearchesJL(; method = LineSearches.Static(), autodiff = nothing, α = true)
    Base.depwarn("`LineSearchesJL(...)` is deprecated. Please use `LineSearchesJL` from \
                  LineSearch.jl instead.",
        :LineSearchesJL)

    # Prevent breaking old code
    method isa LineSearch.LineSearchesJL &&
        return LineSearch.LineSearchesJL(method.method, α, autodiff)
    method isa AbstractLineSearchAlgorithm && return method
    return LineSearch.LineSearchesJL(method, α, autodiff)
end

for alg in (:Static, :HagerZhang, :MoreThuente, :BackTracking, :StrongWolfe)
    depmsg = "`$(alg)(args...; kwargs...)` is deprecated. Please use `LineSearchesJL(; \
              method = $(alg)(args...; kwargs...))` instead."
    @eval function $(alg)(args...; autodiff = nothing, initial_alpha = true, kwargs...)
        Base.depwarn($(depmsg), $(Meta.quot(alg)))
        return LineSearch.LineSearchesJL(;
            method = LineSearches.$(alg)(args...; kwargs...), autodiff, initial_alpha)
    end
end

Base.@deprecate LiFukushimaLineSearch(; nan_max_iter::Int = 5, kwargs...) LineSearch.LiFukushimaLineSearch(;
    nan_maxiters = nan_max_iter, kwargs...)

function callback_into_cache!(topcache, cache::AbstractLineSearchCache, args...)
    LineSearch.callback_into_cache!(cache, get_fu(topcache))
end
