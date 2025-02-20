# Remake the nonlinear problem, then update
function perform_step!(integrator, cache::IDSolveCache, repeat_step = false)
    @unpack alg, u, uprev, dt, t, f, p = integrator
    nlsolve = alg.nlsolve
    @unpack state, prob = cache
    state.u .= uprev
    state.t_next = t
    @show state
    prob = remake(prob, p = state)

    u = solve(prob, nlsolve)
    any(isnan, u) && (integrator.sol.retcode = SciMLBase.ReturnCode.Failure)
    integrator.u = u
end

function initialize!(integrator, cache::IDSolveCache)
    cache.state.u .= integrator.u
    cache.state.p .= integrator.p
    cache.state.t_next = integrator.t
    f = integrator.f

    _f = if isinplace(f)
        (resid, u_next, p) -> f(resid, u_next, p.u, p.p, p.t_next)
    else
        (u_next, p) -> f(u_next, p.u, p.p, p.t_next)
    end

    prob = if isinplace(f)
        NonlinearProblem{true}(_f, cache.state.u, cache.state)
    else
        NonlinearProblem{false}(_f, cache.state.u, cache.state)
    end
    cache.prob = prob
end

#### unnecessary
#function DiffEqBase.__init(prob::ImplicitDiscreteProblem, alg) 
#    f = prob.f
#    t_i = prob.tspan[1]
#    u0 = state_values(prob)
#    p = parameter_values(prob)
#
#    _f(resid, u_next, p) = f(resid, u_next, p.u, p.p, p.t)
#    state = ImplicitDiscreteState(u0, p, t_i)
#    nlprob = NonlinearProblem(_f, u0, state)
#end
