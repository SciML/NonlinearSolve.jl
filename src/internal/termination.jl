function check_and_update!(cache, fu, u, uprev)
    return check_and_update!(cache.termination_cache, cache, fu, u, uprev)
end

function check_and_update!(tc_cache, cache, fu, u, uprev)
    return check_and_update!(tc_cache, cache, fu, u, uprev, tc_cache.mode)
end

function check_and_update!(tc_cache, cache, fu, u, uprev, mode)
    if tc_cache(fu, u, uprev)
        cache.retcode = tc_cache.retcode
        update_from_termination_cache!(tc_cache, cache, mode, u)
        cache.force_stop = true
    end
end

function update_from_termination_cache!(tc_cache, cache, u = get_u(cache))
    return update_from_termination_cache!(tc_cache, cache, tc_cache.mode, u)
end

function update_from_termination_cache!(
        tc_cache, cache, ::AbstractNonlinearTerminationMode, u = get_u(cache))
    evaluate_f!(cache, u, cache.p)
end

function update_from_termination_cache!(
        tc_cache, cache, ::AbstractSafeBestNonlinearTerminationMode, u = get_u(cache))
    if isinplace(cache)
        copyto!(get_u(cache), tc_cache.u)
    else
        set_u!(cache, tc_cache.u)
    end
    evaluate_f!(cache, get_u(cache), cache.p)
end
