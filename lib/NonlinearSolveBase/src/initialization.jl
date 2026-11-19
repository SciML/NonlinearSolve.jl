struct NonlinearSolveDefaultInit <: SciMLBase.DAEInitializationAlgorithm end

"""
    run_initialization!(cache, initializealg = cache.initializealg, prob = cache.prob;
        abstol = nothing, reltol = nothing)

Run `initializealg` on `prob` and write the initialized values into `cache`.

`cache` may be the problem itself when initialization runs before a solver cache exists
(polyalgorithm `solve`, multistart, bounds transform). Pass the `abstol`/`reltol` given to
`solve` in that case: a problem only carries the tolerances stored in `prob.kwargs`, so
without them initialization falls back to the default tolerances. `nothing` uses
`get_abstol(cache)`/`get_reltol(cache)`.
"""
function run_initialization!(
        cache, initializealg = cache.initializealg, prob = cache.prob;
        abstol = nothing, reltol = nothing
    )
    return _run_initialization!(
        cache, initializealg, prob, SciMLBase.isinplace(cache); abstol, reltol
    )
end

function _run_initialization!(
        cache, ::NonlinearSolveDefaultInit, prob, isinplace::Bool; kwargs...
    )
    if SciMLBase.has_initialization_data(prob.f) &&
            prob.f.initialization_data isa SciMLBase.OverrideInitData
        return _run_initialization!(cache, SciMLBase.OverrideInit(), prob, isinplace; kwargs...)
    end
    return cache, true
end

function _run_initialization!(
        cache, initalg::SciMLBase.OverrideInit, prob,
        isinplace::Bool; abstol = nothing, reltol = nothing
    )
    if cache isa AbstractNonlinearSolveCache && isdefined(cache.alg, :autodiff)
        autodiff = cache.alg.autodiff
    else
        autodiff = ADTypes.AutoForwardDiff()
    end
    alg = initialization_alg(prob.f.initialization_data.initializeprob, autodiff)
    if alg === nothing && cache isa AbstractNonlinearSolveCache
        alg = cache.alg
    end
    u0, p, success = SciMLBase.get_initial_values(
        prob, cache, prob.f, initalg, Val(isinplace); nlsolve_alg = alg,
        abstol = abstol === nothing ? get_abstol(cache) : abstol,
        reltol = reltol === nothing ? get_reltol(cache) : reltol
    )
    cache = update_initial_values!(cache, u0, p)
    if cache isa AbstractNonlinearSolveCache && isdefined(cache, :retcode) && !success
        cache.retcode = ReturnCode.InitialFailure
    end

    return cache, success
end

function get_abstol(prob::AbstractNonlinearProblem)
    return get_tolerance(get(prob.kwargs, :abstol, nothing), eltype(SII.state_values(prob)))
end
function get_reltol(prob::AbstractNonlinearProblem)
    return get_tolerance(get(prob.kwargs, :reltol, nothing), eltype(SII.state_values(prob)))
end

"""
    initialization_alg(initprob, autodiff)

Select a nonlinear algorithm for the auxiliary problem in `OverrideInitData`.
Solver packages may extend this hook for supported problem types and should preserve
`initprob`'s constraints while forwarding the requested differentiation backend.

Return `nothing` to leave the algorithm unspecified. When initialization is running on
an existing nonlinear solver cache, this falls back to that cache's algorithm.
"""
initialization_alg(initprob, autodiff) = nothing

function update_initial_values!(cache::AbstractNonlinearSolveCache, u0, p)
    InternalAPI.reinit!(cache; u = get_u(cache), u0, p)
    cache.prob = SciMLBase.remake(cache.prob; u0, p)
    return cache
end

function update_initial_values!(prob::AbstractNonlinearProblem, u0, p)
    return SciMLBase.remake(prob; u0, p)
end

function _run_initialization!(
        cache::Union{AbstractNonlinearSolveCache, AbstractNonlinearProblem}, ::SciMLBase.NoInit, prob, isinplace;
        kwargs...
    )
    return cache, true
end
