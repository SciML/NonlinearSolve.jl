# Some algorithms don't support creating a cache and doing `solve!`, this unfortunately
# makes it difficult to write generic code that supports caching. For the algorithms that
# don't have a `__init` function defined, we create a "Fake Cache", which just calls
# `__solve` from `solve!`
@concrete mutable struct NonlinearSolveNoInitCache{iip, timeit} <:
                         AbstractNonlinearSolveCache{iip, timeit}
    prob
    alg
    args
    kwargs::Any
end

function SciMLBase.reinit!(
        cache::NonlinearSolveNoInitCache, u0 = cache.prob.u0; p = cache.prob.p, kwargs...)
    prob = remake(cache.prob; u0, p)
    cache.prob = prob
    cache.kwargs = merge(cache.kwargs, kwargs)
    return cache
end

function Base.show(io::IO, cache::NonlinearSolveNoInitCache)
    print(io, "NonlinearSolveNoInitCache(alg = $(cache.alg))")
end

function SciMLBase.__init(prob::AbstractNonlinearProblem{uType, iip},
        alg::Union{AbstractNonlinearSolveAlgorithm,
            SimpleNonlinearSolve.AbstractSimpleNonlinearSolveAlgorithm},
        args...;
        maxtime = nothing,
        kwargs...) where {uType, iip}
    return NonlinearSolveNoInitCache{iip, maxtime !== nothing}(
        prob, alg, args, merge((; maxtime), kwargs))
end

function SciMLBase.solve!(cache::NonlinearSolveNoInitCache)
    return solve(cache.prob, cache.alg, cache.args...; cache.kwargs...)
end
