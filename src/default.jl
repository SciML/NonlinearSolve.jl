"""
    RobustMultiNewton(; concrete_jac = nothing, linsolve = nothing, precs = DEFAULT_PRECS,
                        adkwargs...)

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

@concrete mutable struct RobustMultiNewtonCache{iip, N} <: AbstractNonlinearSolveCache{iip}
    caches
    alg
    current::Int
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::RobustMultiNewton,
    args...; kwargs...) where {uType, iip}
    @unpack adkwargs, linsolve, precs = alg

    algs = (TrustRegion(; linsolve, precs, adkwargs...),
        TrustRegion(; linsolve, precs,
            radius_update_scheme = RadiusUpdateSchemes.Bastin, adkwargs...),
        NewtonRaphson(; linsolve, precs, linesearch = BackTracking(), adkwargs...),
        TrustRegion(; linsolve, precs,
            radius_update_scheme = RadiusUpdateSchemes.NLsolve, adkwargs...),
        TrustRegion(; linsolve, precs,
            radius_update_scheme = RadiusUpdateSchemes.Fan, adkwargs...))

    # Partially Type Unstable but can't do much since some upstream caches -- LineSearches
    # and SparseDiffTools cause the instability
    return RobustMultiNewtonCache{iip, length(algs)}(map(solver -> SciMLBase.__init(prob,
                solver, args...; kwargs...), algs), alg, 1)
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
    return FastShortcutNonlinearPolyalg{_unwrap_val(concrete_jac)}(adkwargs, linsolve,
        precs)
end

@concrete mutable struct FastShortcutNonlinearPolyalgCache{iip, N} <:
                         AbstractNonlinearSolveCache{iip}
    caches
    alg
    current::Int
end

function FastShortcutNonlinearPolyalgCache(; concrete_jac = nothing, linsolve = nothing,
    precs = DEFAULT_PRECS, adkwargs...)
    return FastShortcutNonlinearPolyalgCache{_unwrap_val(concrete_jac)}(adkwargs, linsolve,
        precs)
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip},
    alg::FastShortcutNonlinearPolyalg, args...; kwargs...) where {uType, iip}
    @unpack adkwargs, linsolve, precs = alg

    algs = (
        # Klement(),
        # Broyden(),
        NewtonRaphson(; linsolve, precs, adkwargs...),
        NewtonRaphson(; linsolve, precs, linesearch = BackTracking(), adkwargs...),
        TrustRegion(; linsolve, precs, adkwargs...),
        TrustRegion(; linsolve, precs,
            radius_update_scheme = RadiusUpdateSchemes.Bastin, adkwargs...))

    return FastShortcutNonlinearPolyalgCache{iip, length(algs)}(map(solver -> SciMLBase.__init(prob,
                solver, args...; kwargs...), algs), alg, 1)
end

# This version doesn't allocate all the caches!
@generated function SciMLBase.__solve(prob::NonlinearProblem{uType, iip},
    alg::Union{FastShortcutNonlinearPolyalg, RobustMultiNewton}, args...;
    kwargs...) where {uType, iip}
    calls = [:(@unpack adkwargs, linsolve, precs = alg)]

    algs = if parameterless_type(alg) == RobustMultiNewton
        [
            :(TrustRegion(; linsolve, precs, adkwargs...)),
            :(TrustRegion(; linsolve, precs,
                radius_update_scheme = RadiusUpdateSchemes.Bastin, adkwargs...)),
            :(NewtonRaphson(; linsolve, precs, linesearch = BackTracking(), adkwargs...)),
            :(TrustRegion(; linsolve, precs,
                radius_update_scheme = RadiusUpdateSchemes.NLsolve, adkwargs...)),
            :(TrustRegion(; linsolve, precs,
                radius_update_scheme = RadiusUpdateSchemes.Fan, adkwargs...)),
        ]
    else
        [
            :(GeneralKlement()),
            :(GeneralBroyden()),
            :(NewtonRaphson(; linsolve, precs, adkwargs...)),
            :(NewtonRaphson(; linsolve, precs, linesearch = BackTracking(), adkwargs...)),
            :(TrustRegion(; linsolve, precs, adkwargs...)),
            :(TrustRegion(; linsolve, precs,
                radius_update_scheme = RadiusUpdateSchemes.Bastin, adkwargs...)),
        ]
    end
    filter!(!isnothing, algs)
    sol_syms = [gensym("sol") for i in 1:length(algs)]
    for i in 1:length(algs)
        cur_sol = sol_syms[i]
        push!(calls,
            quote
                $(cur_sol) = SciMLBase.__solve(prob, $(algs[i]), args...; kwargs...)
                if SciMLBase.successful_retcode($(cur_sol))
                    return SciMLBase.build_solution(prob, alg, $(cur_sol).u,
                        $(cur_sol).resid; $(cur_sol).retcode, $(cur_sol).stats,
                        original = $(cur_sol))
                end
            end)
    end

    resids = map(x -> Symbol("$(x)_resid"), sol_syms)
    for (sym, resid) in zip(sol_syms, resids)
        push!(calls, :($(resid) = $(sym).resid))
    end

    push!(calls,
        quote
            resids = tuple($(Tuple(resids)...))
            minfu, idx = findmin(DEFAULT_NORM, resids)
        end)

    for i in 1:length(algs)
        push!(calls,
            quote
                if idx == $i
                    return SciMLBase.build_solution(prob, alg, $(sol_syms[i]).u,
                        $(sol_syms[i]).resid; $(sol_syms[i]).retcode, $(sol_syms[i]).stats)
                end
            end)
    end
    push!(calls, :(error("Current choices shouldn't get here!")))

    return Expr(:block, calls...)
end

## General shared polyalg functions

@generated function SciMLBase.solve!(cache::Union{RobustMultiNewtonCache{iip, N},
    FastShortcutNonlinearPolyalgCache{iip, N}}) where {iip, N}
    calls = [
        quote
            1 ≤ cache.current ≤ length(cache.caches) ||
                error("Current choices shouldn't get here!")
        end,
    ]

    cache_syms = [gensym("cache") for i in 1:N]
    sol_syms = [gensym("sol") for i in 1:N]
    for i in 1:N
        push!(calls,
            quote
                $(cache_syms[i]) = cache.caches[$(i)]
                if $(i) == cache.current
                    $(sol_syms[i]) = SciMLBase.solve!($(cache_syms[i]))
                    if SciMLBase.successful_retcode($(sol_syms[i]))
                        stats = $(sol_syms[i]).stats
                        u = $(sol_syms[i]).u
                        fu = get_fu($(cache_syms[i]))
                        return SciMLBase.build_solution($(sol_syms[i]).prob, cache.alg, u,
                            fu; retcode = ReturnCode.Success, stats,
                            original = $(sol_syms[i]))
                    end
                    cache.current = $(i + 1)
                end
            end)
    end

    resids = map(x -> Symbol("$(x)_resid"), cache_syms)
    for (sym, resid) in zip(cache_syms, resids)
        push!(calls, :($(resid) = get_fu($(sym))))
    end
    push!(calls,
        quote
            retcode = ReturnCode.MaxIters

            fus = tuple($(Tuple(resids)...))
            minfu, idx = findmin(cache.caches[1].internalnorm, fus)
            stats = cache.caches[idx].stats
            u = cache.caches[idx].u

            return SciMLBase.build_solution(cache.caches[idx].prob, cache.alg, u,
                fus[idx]; retcode, stats)
        end)

    return Expr(:block, calls...)
end

function SciMLBase.reinit!(cache::Union{RobustMultiNewtonCache,
        FastShortcutNonlinearPolyalgCache}, args...; kwargs...)
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
