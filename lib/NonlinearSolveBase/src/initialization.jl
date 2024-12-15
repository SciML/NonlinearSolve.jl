struct NonlinearSolveDefaultInit <: SciMLBase.DAEInitializationAlgorithm end

function run_initialization!(cache, initializealg = cache.initializealg, prob = cache.prob)
    _run_initialization!(cache, initializealg, prob, Val(SciMLBase.isinplace(cache)))
end

function _run_initialization!(
        cache, ::NonlinearSolveDefaultInit, prob, isinplace::Union{Val{true}, Val{false}})
    if SciMLBase.has_initialization_data(prob.f) &&
       prob.f.initialization_data isa SciMLBase.OverrideInitData
        return _run_initialization!(cache, SciMLBase.OverrideInit(), prob, isinplace)
    end
    return cache, true
end

function _run_initialization!(cache, initalg::SciMLBase.OverrideInit, prob,
        isinplace::Union{Val{true}, Val{false}})
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
        prob, cache, prob.f, initalg, isinplace; nlsolve_alg = alg,
        abstol = get_abstol(cache), reltol = get_reltol(cache))
    cache = update_initial_values!(cache, u0, p)
    if cache isa AbstractNonlinearSolveCache && isdefined(cache, :retcode) && !success
        cache.retcode = ReturnCode.InitialFailure
    end

    return cache, success
end

function get_abstol(prob::AbstractNonlinearProblem)
    get_tolerance(get(prob.kwargs, :abstol, nothing), eltype(SII.state_values(prob)))
end
function get_reltol(prob::AbstractNonlinearProblem)
    get_tolerance(get(prob.kwargs, :reltol, nothing), eltype(SII.state_values(prob)))
end

initialization_alg(initprob, autodiff) = nothing

function update_initial_values!(cache::AbstractNonlinearSolveCache, u0, p)
    InternalAPI.reinit!(cache; u0, p)
    cache.prob = SciMLBase.remake(cache.prob; u0, p)
    return cache
end

function update_initial_values!(prob::AbstractNonlinearProblem, u0, p)
    return SciMLBase.remake(prob; u0, p)
end

function _run_initialization!(
        cache::AbstractNonlinearSolveCache, ::SciMLBase.NoInit, prob, isinplace)
    return cache, true
end
