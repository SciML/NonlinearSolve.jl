struct NonlinearSolveDefaultInit <: SciMLBase.DAEInitializationAlgorithm end

function initialize_cache!(cache::AbstractNonlinearSolveCache, initializealg = cache.initializealg, prob = cache.prob)
    _initialize_cache!(cache, initializealg, prob, Val(SciMLBase.isinplace(cache)))
end

function _initialize_cache!(cache::AbstractNonlinearSolveCache, ::NonlinearSolveDefaultInit, prob, isinplace::Union{Val{true}, Val{false}})
    if SciMLBase.has_initialization_data(prob.f)
        _initialize_cache!(cache, SciMLBase.OverrideInit(), prob, isinplace)
    end
end

function _initialize_cache!(cache::AbstractNonlinearSolveCache, initalg::SciMLBase.OverrideInit, prob, isinplace::Union{Val{true}, Val{false}})
    initprob = prob.f.initialization_data.initializeprob

    u0, p, success = SciMLBase.get_initial_values(prob, cache, prob.f, initalg, isinplace; nlsolve_alg = initialization_alg(cache), abstol = get_abstol(cache), reltol = get_reltol(cache))
    SciMLBase.set_u!(cache, u0)
    update_parameter_object!(cache, p)
    if !success
        cache.retcode = ReturnCode.InitialFailure
    end
end

initialization_alg(cache::AbstractNonlinearSolveCache) = cache.alg
function update_parameter_object!(cache::AbstractNonlinearSolveCache, p)
    cache.p = p
end

function _initialize_cache!(cache::AbstractNonlinearSolveCache, ::SciMLBase.NoInit, prob, isinplace) end
