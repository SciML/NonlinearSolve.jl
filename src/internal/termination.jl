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
        # Just a sanity measure!
        if isinplace(cache)
            cache.prob.f(get_fu(cache), u, cache.prob.p)
        else
            set_fu!(cache, cache.prob.f(u, cache.prob.p))
        end
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
        # Just a sanity measure!
        if isinplace(cache)
            cache.prob.f(get_fu(cache), u, cache.prob.p)
        else
            set_fu!(cache, cache.prob.f(u, cache.prob.p))
        end
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
        if isinplace(cache)
            copyto!(get_u(cache), tc_cache.u)
            cache.prob.f(get_fu(cache), get_u(cache), cache.prob.p)
        else
            set_u!(cache, tc_cache.u)
            set_fu!(cache, cache.prob.f(get_u(cache), cache.prob.p))
        end
        cache.force_stop = true
    end
end