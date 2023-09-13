mutable struct DFSaneCache{iip}
    f::fType
    alg::algType
    u::uType
    fu::resType
    p::pType
    uf::ufType
    du_tmp::duType
    force_stop::Bool
    maxiters::Int
    internalnorm::INType
    retcode::SciMLBase.ReturnCode.T
    abstol::tolType
    prob::probType
    stats::NLStats

    function DFSaneCache()
    end
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::DFSane)
    return DFSaneCache()
end

function perform_step!(cache::DFSaneCache{true})
    return nothing
end

function perform_step!(cache::DFSaneCache{false})
    return nothing
end

function SciMLBase.solve!(cache::DFSaneCache)
    while !cache.force_stop && cache.stats.nsteps < cache.maxiters
        perform_step!(cache)
        cache.stats.nsteps += 1
    end

    if cache.stats.nsteps == cache.maxiters
        cache.retcode = ReturnCode.MaxIters
    else
        cache.retcode = ReturnCode.Success
    end

    SciMLBase.build_solution(cache.prob, cache.alg, cache.u, cache.fu;
        retcode = cache.retcode, stats = cache.stats)
end
