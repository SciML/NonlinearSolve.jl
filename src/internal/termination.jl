function init_termination_cache(abstol, reltol, du, u, ::Nothing)
    return init_termination_cache(abstol, reltol, du, u, AbsSafeBestTerminationMode())
end
function init_termination_cache(abstol, reltol, du, u, tc::AbstractNonlinearTerminationMode)
    tc_cache = init(du, u, tc; abstol, reltol)
    return DiffEqBase.get_abstol(tc_cache), DiffEqBase.get_reltol(tc_cache), tc_cache
end

function check_and_update!(cache, fu, u, uprev)
    return check_and_update!(cache.termination_cache, cache, fu, u, uprev)
end

function check_and_update!(tc_cache, cache, fu, u, uprev)
    return check_and_update!(tc_cache, cache, fu, u, uprev,
        DiffEqBase.get_termination_mode(tc_cache))
end

# FIXME: The return codes need to synced up with SciMLBase.ReturnCode
function check_and_update!(tc_cache, cache, fu, u, uprev,
        mode::AbstractNonlinearTerminationMode)
    if tc_cache(fu, u, uprev)
        update_from_termination_cache!(tc_cache, cache, mode, u)
        cache.force_stop = true
    end
end

function check_and_update!(tc_cache, cache, fu, u, uprev,
        mode::AbstractSafeNonlinearTerminationMode)
    if tc_cache(fu, u, uprev)
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.Success
            cache.retcode = ReturnCode.Success
        end
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.PatienceTermination
            cache.retcode = ReturnCode.ConvergenceFailure
        end
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.ProtectiveTermination
            cache.retcode = ReturnCode.Unstable
        end
        update_from_termination_cache!(tc_cache, cache, mode, u)
        cache.force_stop = true
    end
end

function check_and_update!(tc_cache, cache, fu, u, uprev,
        mode::AbstractSafeBestNonlinearTerminationMode)
    if tc_cache(fu, u, uprev)
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.Success
            cache.retcode = ReturnCode.Success
        end
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.PatienceTermination
            cache.retcode = ReturnCode.ConvergenceFailure
        end
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.ProtectiveTermination
            cache.retcode = ReturnCode.Unstable
        end
        update_from_termination_cache!(tc_cache, cache, mode, u)
        cache.force_stop = true
    end
end

function update_from_termination_cache!(tc_cache, cache, u = get_u(cache))
    return update_from_termination_cache!(tc_cache, cache,
        DiffEqBase.get_termination_mode(tc_cache), u)
end

function update_from_termination_cache!(tc_cache, cache,
        mode::AbstractNonlinearTerminationMode, u = get_u(cache))
    evaluate_f!(cache, u, cache.p)
end

function update_from_termination_cache!(tc_cache, cache,
        mode::AbstractSafeBestNonlinearTerminationMode, u = get_u(cache))
    if isinplace(cache)
        copyto!(get_u(cache), tc_cache.u)
    else
        set_u!(cache, tc_cache.u)
    end
    evaluate_f!(cache, get_u(cache), cache.p)
end
