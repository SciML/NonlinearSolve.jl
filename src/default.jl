"""
    NonlinearSolvePolyAlgorithm(algs, ::Val{pType} = Val(:NLS)) where {pType}

A general way to define PolyAlgorithms for `NonlinearProblem` and
`NonlinearLeastSquaresProblem`. This is a container for a tuple of algorithms that will be
tried in order until one succeeds. If none succeed, then the algorithm with the lowest
residual is returned.

## Arguments

  - `algs`: a tuple of algorithms to try in-order! (If this is not a Tuple, then the
    returned algorithm is not type-stable).
  - `pType`: the problem type. Defaults to `:NLS` for `NonlinearProblem` and `:NLLS` for
    `NonlinearLeastSquaresProblem`. This is used to determine the correct problem type to
    dispatch on.

## Example

```julia
using NonlinearSolve

alg = NonlinearSolvePolyAlgorithm((NewtonRaphson(), GeneralBroyden()))
```
"""
struct NonlinearSolvePolyAlgorithm{pType, N, A} <: AbstractNonlinearSolveAlgorithm
    algs::A

    function NonlinearSolvePolyAlgorithm(algs, ::Val{pType} = Val(:NLS)) where {pType}
        @assert pType ∈ (:NLS, :NLLS)
        algs = Tuple(algs)
        return new{pType, length(algs), typeof(algs)}(algs)
    end
end

function Base.show(io::IO, alg::NonlinearSolvePolyAlgorithm{pType, N}) where {pType, N}
    problem_kind = ifelse(pType == :NLS, "NonlinearProblem", "NonlinearLeastSquaresProblem")
    println(io, "NonlinearSolvePolyAlgorithm for $(problem_kind) with $(N) algorithms")
    for i in 1:(N - 1)
        println(io, "  $(i): $(alg.algs[i])")
    end
    print(io, "  $(N): $(alg.algs[N])")
end

@concrete mutable struct NonlinearSolvePolyAlgorithmCache{iip, N} <:
                         AbstractNonlinearSolveCache{iip}
    caches
    alg
    current::Int
end

for (probType, pType) in ((:NonlinearProblem, :NLS), (:NonlinearLeastSquaresProblem, :NLLS))
    algType = NonlinearSolvePolyAlgorithm{pType}
    @eval begin
        function SciMLBase.__init(prob::$probType, alg::$algType{N}, args...;
                kwargs...) where {N}
            return NonlinearSolvePolyAlgorithmCache{isinplace(prob), N}(map(solver -> SciMLBase.__init(prob,
                        solver, args...; kwargs...), alg.algs), alg, 1)
        end
    end
end

@generated function SciMLBase.solve!(cache::NonlinearSolvePolyAlgorithmCache{iip,
        N}) where {iip, N}
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
            minfu, idx = __findmin(cache.caches[1].internalnorm, fus)
            stats = cache.caches[idx].stats
            u = cache.caches[idx].u

            return SciMLBase.build_solution(cache.caches[idx].prob, cache.alg, u,
                fus[idx]; retcode, stats)
        end)

    return Expr(:block, calls...)
end

for (probType, pType) in ((:NonlinearProblem, :NLS), (:NonlinearLeastSquaresProblem, :NLLS))
    algType = NonlinearSolvePolyAlgorithm{pType}
    @eval begin
        @generated function SciMLBase.__solve(prob::$probType, alg::$algType{N}, args...;
                kwargs...) where {N}
            calls = []
            sol_syms = [gensym("sol") for _ in 1:N]
            for i in 1:N
                cur_sol = sol_syms[i]
                push!(calls,
                    quote
                        $(cur_sol) = SciMLBase.__solve(prob, alg.algs[$(i)], args...;
                            kwargs...)
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
                    minfu, idx = __findmin(DEFAULT_NORM, resids)
                end)

            for i in 1:N
                push!(calls,
                    quote
                        if idx == $i
                            return SciMLBase.build_solution(prob, alg, $(sol_syms[i]).u,
                                $(sol_syms[i]).resid; $(sol_syms[i]).retcode,
                                $(sol_syms[i]).stats)
                        end
                    end)
            end
            push!(calls, :(error("Current choices shouldn't get here!")))

            return Expr(:block, calls...)
        end
    end
end

function SciMLBase.reinit!(cache::NonlinearSolvePolyAlgorithmCache, args...; kwargs...)
    for c in cache.caches
        SciMLBase.reinit!(c, args...; kwargs...)
    end
end

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
function RobustMultiNewton(; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, adkwargs...)
    algs = (TrustRegion(; concrete_jac, linsolve, precs, adkwargs...),
        TrustRegion(; concrete_jac, linsolve, precs,
            radius_update_scheme = RadiusUpdateSchemes.Bastin, adkwargs...),
        NewtonRaphson(; concrete_jac, linsolve, precs, linesearch = BackTracking(),
            adkwargs...),
        TrustRegion(; concrete_jac, linsolve, precs,
            radius_update_scheme = RadiusUpdateSchemes.NLsolve, adkwargs...),
        TrustRegion(; concrete_jac, linsolve, precs,
            radius_update_scheme = RadiusUpdateSchemes.Fan, adkwargs...))
    return NonlinearSolvePolyAlgorithm(algs, Val(:NLS))
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
function FastShortcutNonlinearPolyalg(; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, adkwargs...)
    algs = (GeneralKlement(; linsolve, precs),
        GeneralBroyden(),
        NewtonRaphson(; concrete_jac, linsolve, precs, adkwargs...),
        NewtonRaphson(; concrete_jac, linsolve, precs, linesearch = BackTracking(),
            adkwargs...),
        TrustRegion(; concrete_jac, linsolve, precs, adkwargs...),
        TrustRegion(; concrete_jac, linsolve, precs,
            radius_update_scheme = RadiusUpdateSchemes.Bastin, adkwargs...))
    return NonlinearSolvePolyAlgorithm(algs, Val(:NLS))
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
