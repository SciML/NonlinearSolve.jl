"""
RobustMultiNewton(; concrete_jac = nothing, linsolve = nothing,
                    precs = DEFAULT_PRECS, adkwargs...)

A polyalgorithm focused on robustness. It uses a mixture of Newton methods with different
globalizing techniques (trust region updates, line searches, etc.) in order to find a
method that is able to adequately solve the minimization problem.

Basically, if this algorithm fails, then "most" good ways of solving your problem fail and
you may need to think about reformulating the model (either there is an issue with the model,
or more precision / more stable linear solver choice is required).

### Keyword Arguments

- `autodiff`: determines the backend used for the Jacobian. Note that this argument is
    ignored if an analytical Jacobian is passed, as that will be used instead. Defaults to
    `AutoForwardDiff()`. Valid choices are types from ADTypes.jl.
- `concrete_jac`: whether to build a concrete Jacobian. If a Krylov-subspace method is used,
    then the Jacobian will not be constructed and instead direct Jacobian-vector products
    `J*v` are computed using forward-mode automatic differentiation or finite differencing
    tricks (without ever constructing the Jacobian). However, if the Jacobian is still needed,
    for example for a preconditioner, `concrete_jac = true` can be passed in order to force
    the construction of the Jacobian.
- `linsolve`: the [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) used for the
    linear solves within the Newton method. Defaults to `nothing`, which means it uses the
    LinearSolve.jl default algorithm choice. For more information on available algorithm
    choices, see the [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
- `precs`: the choice of preconditioners for the linear solver. Defaults to using no
    preconditioners. For more information on specifying preconditioners for LinearSolve
    algorithms, consult the
    [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
"""
@concrete struct RobustMultiNewton{CJ} <: AbstractNewtonAlgorithm{CJ, Nothing}
    adkwargs
    linsolve
    precs
end

# When somethin's strange, and numerical
# who you gonna call?
# Robusters!
const Robusters = RobustMultiNewton

function RobustMultiNewton(; concrete_jac = nothing, linsolve = nothing, 
        precs = DEFAULT_PRECS, adkwargs...)

    return RobustMultiNewton{_unwrap_val(concrete_jac)}(adkwargs, linsolve, precs)
end

@concrete mutable struct RobustMultiNewtonCache{iip} <: AbstractNonlinearSolveCache{iip}
    caches
    alg
    current::Int
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::RobustMultiNewton, args...; 
    kwargs...) where {uType, iip}

    adkwargs = alg.adkwargs
    linsolve = alg.linsolve
    precs = alg.precs

    RobustMultiNewtonCache{iip}((
        SciMLBase.__init(prob, TrustRegion(;linsolve, precs, adkwargs...), args...; kwargs...),
        SciMLBase.__init(prob, TrustRegion(;linsolve, precs, radius_update_scheme = RadiusUpdateSchemes.Bastin, adkwargs...), args...; kwargs...),
        SciMLBase.__init(prob, NewtonRaphson(;linsolve, precs, linesearch=BackTracking(), adkwargs...), args...; kwargs...),
        SciMLBase.__init(prob, TrustRegion(;linsolve, precs, radius_update_scheme = RadiusUpdateSchemes.Fan, adkwargs...), args...; kwargs...),
        ), alg, 1
    )
end

"""
FastShortcutNonlinearPolyalg(; concrete_jac = nothing, linsolve = nothing,
                    precs = DEFAULT_PRECS, adkwargs...)

A polyalgorithm focused on balancing speed and robustness. It first tries less robust methods
for more performance and then tries more robust techniques if the faster ones fail.

### Keyword Arguments

- `autodiff`: determines the backend used for the Jacobian. Note that this argument is
    ignored if an analytical Jacobian is passed, as that will be used instead. Defaults to
    `AutoForwardDiff()`. Valid choices are types from ADTypes.jl.
- `concrete_jac`: whether to build a concrete Jacobian. If a Krylov-subspace method is used,
    then the Jacobian will not be constructed and instead direct Jacobian-vector products
    `J*v` are computed using forward-mode automatic differentiation or finite differencing
    tricks (without ever constructing the Jacobian). However, if the Jacobian is still needed,
    for example for a preconditioner, `concrete_jac = true` can be passed in order to force
    the construction of the Jacobian.
- `linsolve`: the [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) used for the
    linear solves within the Newton method. Defaults to `nothing`, which means it uses the
    LinearSolve.jl default algorithm choice. For more information on available algorithm
    choices, see the [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
- `precs`: the choice of preconditioners for the linear solver. Defaults to using no
    preconditioners. For more information on specifying preconditioners for LinearSolve
    algorithms, consult the
    [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
"""
@concrete struct FastShortcutNonlinearPolyalg{CJ} <: AbstractNewtonAlgorithm{CJ, Nothing}
    adkwargs
    linsolve
    precs
end

function FastShortcutNonlinearPolyalg(; concrete_jac = nothing, linsolve = nothing, 
    precs = DEFAULT_PRECS, adkwargs...)

    return FastShortcutNonlinearPolyalg{_unwrap_val(concrete_jac)}(adkwargs, linsolve, precs)
end

@concrete mutable struct FastShortcutNonlinearPolyalgCache{iip} <: AbstractNonlinearSolveCache{iip}
    caches
    alg
    current::Int
end

function FastShortcutNonlinearPolyalgCache(; concrete_jac = nothing, linsolve = nothing, 
        precs = DEFAULT_PRECS, adkwargs...)

    return FastShortcutNonlinearPolyalgCache{_unwrap_val(concrete_jac)}(adkwargs, linsolve, precs)
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::FastShortcutNonlinearPolyalg, args...; 
    kwargs...) where {uType, iip}

    adkwargs = alg.adkwargs
    linsolve = alg.linsolve
    precs = alg.precs

    FastShortcutNonlinearPolyalgCache{iip}((
        #SciMLBase.__init(prob, Klement(), args...; kwargs...),
        #SciMLBase.__init(prob, Broyden(), args...; kwargs...),
        SciMLBase.__init(prob, NewtonRaphson(;linsolve, precs, adkwargs...), args...; kwargs...),
        SciMLBase.__init(prob, NewtonRaphson(;linsolve, precs, linesearch=BackTracking(), adkwargs...), args...; kwargs...),
        SciMLBase.__init(prob, TrustRegion(;linsolve, precs, adkwargs...), args...; kwargs...),
        SciMLBase.__init(prob, TrustRegion(;linsolve, precs, radius_update_scheme = RadiusUpdateSchemes.Bastin, adkwargs...), args...; kwargs...),
        ), alg, 1
    )
end

function SciMLBase.__solve(prob::NonlinearProblem{uType, false}, alg::FastShortcutNonlinearPolyalg, args...; 
    kwargs...) where {uType}

    adkwargs = alg.adkwargs
    linsolve = alg.linsolve
    precs = alg.precs

    sol1 = SciMLBase.__solve(prob, Klement(), args...; kwargs...)
    if SciMLBase.successful_retcode(sol1)
        return SciMLBase.build_solution(prob, alg, sol1.u, sol1.resid;
                                        sol1.retcode, sol1.stats)
    end

    sol2 = SciMLBase.__solve(prob, Broyden(), args...; kwargs...)
    if SciMLBase.successful_retcode(sol2)
        return SciMLBase.build_solution(prob, alg, sol2.u, sol2.resid;
                                        sol2.retcode, sol2.stats)
    end

    sol3 = SciMLBase.__solve(prob, NewtonRaphson(;linsolve, precs, adkwargs...), args...; kwargs...)
    if SciMLBase.successful_retcode(sol3)
        return SciMLBase.build_solution(prob, alg, sol3.u, sol3.resid;
                                        sol3.retcode, sol3.stats)
    end

    sol4 = SciMLBase.__solve(prob,  TrustRegion(;linsolve, precs, adkwargs...), args...; kwargs...)
    if SciMLBase.successful_retcode(sol4)
        return SciMLBase.build_solution(prob, alg, sol4.u, sol4.resid;
                                        sol4.retcode, sol4.stats)
    end

    resids = (sol1.resid, sol2.resid, sol3.resid, sol4.resid)
    minfu, idx = findmin(DEFAULT_NORM, resids)

    if idx == 1
        SciMLBase.build_solution(prob, alg, sol1.u, sol1.resid;
                                        sol1.retcode, sol1.stats)
    elseif idx == 2
        SciMLBase.build_solution(prob, alg, sol2.u, sol2.resid;
                                        sol2.retcode, sol2.stats)
    elseif idx == 3
        SciMLBase.build_solution(prob, alg, sol3.u, sol3.resid;
                                sol3.retcode, sol3.stats)
    elseif idx == 4
        SciMLBase.build_solution(prob, alg, sol4.u, sol4.resid;
                                sol4.retcode, sol4.stats)
    else
        error("Unreachable reached, 박정석")
    end

end

function SciMLBase.__solve(prob::NonlinearProblem{uType, true}, alg::FastShortcutNonlinearPolyalg, args...; 
    kwargs...) where {uType}

    adkwargs = alg.adkwargs
    linsolve = alg.linsolve
    precs = alg.precs

    sol1 = SciMLBase.__solve(prob, NewtonRaphson(;linsolve, precs, adkwargs...), args...; kwargs...)
    if SciMLBase.successful_retcode(sol1)
        return SciMLBase.build_solution(prob, alg, sol1.u, sol1.resid;
                                        sol1.retcode, sol1.stats)
    end

    sol2 = SciMLBase.__solve(prob, NewtonRaphson(;linsolve, precs, linesearch=BackTracking(), adkwargs...), args...; kwargs...)
    if SciMLBase.successful_retcode(sol2)
        return SciMLBase.build_solution(prob, alg, sol2.u, sol2.resid;
                                        sol2.retcode, sol2.stats)
    end

    sol3 = SciMLBase.__solve(prob, TrustRegion(;linsolve, precs, adkwargs...), args...; kwargs...)
    if SciMLBase.successful_retcode(sol3)
        return SciMLBase.build_solution(prob, alg, sol3.u, sol3.resid;
                                        sol3.retcode, sol3.stats)
    end

    sol4 = SciMLBase.__solve(prob,  TrustRegion(;linsolve, precs, radius_update_scheme = RadiusUpdateSchemes.Bastin, adkwargs...), args...; kwargs...)
    if SciMLBase.successful_retcode(sol4)
        return SciMLBase.build_solution(prob, alg, sol4.u, sol4.resid;
                                        sol4.retcode, sol4.stats)
    end

    resids = (sol1.resid, sol2.resid, sol3.resid, sol4.resid)
    minfu, idx = findmin(DEFAULT_NORM, resids)

    if idx == 1
        SciMLBase.build_solution(prob, alg, sol1.u, sol1.resid;
                                        sol1.retcode, sol1.stats)
    elseif idx == 2
        SciMLBase.build_solution(prob, alg, sol2.u, sol2.resid;
                                        sol2.retcode, sol2.stats)
    elseif idx == 3
        SciMLBase.build_solution(prob, alg, sol3.u, sol3.resid;
                                sol3.retcode, sol3.stats)
    elseif idx == 4
        SciMLBase.build_solution(prob, alg, sol4.u, sol4.resid;
                                sol4.retcode, sol4.stats)
    else
        error("Unreachable reached, 박정석")
    end

end

## General shared polyalg functions

function perform_step!(cache::Union{RobustMultiNewtonCache, FastShortcutNonlinearPolyalgCache})
    current = cache.current

    while true
        if current == 1
            perform_step!(cache.caches[1])
        elseif current == 2
            perform_step!(cache.caches[2])
        elseif current == 3
            perform_step!(cache.caches[3])
        elseif current == 4
            perform_step!(cache.caches[4])
        else
            error("Current choices shouldn't get here!")
        end
    end

    return nothing
end

function SciMLBase.solve!(cache::Union{RobustMultiNewtonCache, FastShortcutNonlinearPolyalgCache})
    current = cache.current

    while current < 5 && all(not_terminated, cache.caches)
        if current == 1
            perform_step!(cache.caches[1])
            !not_terminated(cache.caches[1]) && (cache.current += 1)
        elseif current == 2
            perform_step!(cache.caches[2])
            !not_terminated(cache.caches[2]) && (cache.current += 1)
        elseif current == 3
            perform_step!(cache.caches[3])
            !not_terminated(cache.caches[3]) && (cache.current += 1)
        elseif current == 4
            perform_step!(cache.caches[4])
            !not_terminated(cache.caches[4]) && (cache.current += 1)
        else
            error("Current choices shouldn't get here!")
        end

        

        #cache.stats.nsteps += 1
    end

    if current < 5
        stats = if current == 1
            cache.caches[1].stats
        elseif current == 2
            cache.caches[2].stats
        elseif current == 3
            cache.caches[3].stats
        elseif current == 4
            cache.caches[4].stats
        end

        u = if current == 1
            cache.caches[1].u
        elseif current == 2
            cache.caches[2].u
        elseif current == 3
            cache.caches[3].u
        elseif current == 4
            cache.caches[4].u
        end

        fu = if current == 1
            get_fu(cache.caches[1])
        elseif current == 2
            get_fu(cache.caches[2])
        elseif current == 3
            get_fu(cache.caches[3])
        elseif current == 4
            get_fu(cache.caches[4])
        end

        retcode = if stats.nsteps == cache.caches[1].maxiters
            ReturnCode.MaxIters
        else
            ReturnCode.Success
        end

        return SciMLBase.build_solution(cache.caches[1].prob, cache.alg, u, fu;
            retcode, stats)
    else
        retcode = ReturnCode.MaxIters

        fus = (get_fu(cache.caches[1]), get_fu(cache.caches[2]), get_fu(cache.caches[3]), get_fu(cache.caches[4]))
        minfu, idx = findmin(cache.caches[1].internalnorm, fus)

        stats = if idx == 1
            cache.caches[1].stats
        elseif idx == 2
            cache.caches[2].stats
        elseif idx == 3
            cache.caches[3].stats
        elseif idx == 4
            cache.caches[4].stats
        end

        u = if idx == 1
            cache.caches[1].u
        elseif idx == 2
            cache.caches[2].u
        elseif idx == 3
            cache.caches[3].u
        elseif idx == 4
            cache.caches[4].u
        end
        
        return SciMLBase.build_solution(cache.caches[1].prob, cache.alg, u, fu;
                                        retcode, stats)
    end
end

function SciMLBase.reinit!(cache::Union{RobustMultiNewtonCache, FastShortcutNonlinearPolyalgCache}, args...; kwargs...)
    for c in cache.caches
        SciMLBase.reinit!(c, args...; kwargs...)
    end
end

## Defaults

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::Nothing, args...; 
    kwargs...) where {uType, iip}

    SciMLBase.__init(prob, FastShortcutNonlinearPolyalg(), args...; kwargs...)
end

function SciMLBase.__solve(prob::NonlinearProblem{uType, iip}, alg::Nothing, args...; 
    kwargs...) where {uType, iip}

    SciMLBase.__solve(prob, FastShortcutNonlinearPolyalg(), args...; kwargs...)
end