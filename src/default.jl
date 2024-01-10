# Poly Algorithms
"""
    NonlinearSolvePolyAlgorithm(algs, ::Val{pType} = Val(:NLS)) where {pType}

A general way to define PolyAlgorithms for `NonlinearProblem` and
`NonlinearLeastSquaresProblem`. This is a container for a tuple of algorithms that will be
tried in order until one succeeds. If none succeed, then the algorithm with the lowest
residual is returned.

### Arguments

  - `algs`: a tuple of algorithms to try in-order! (If this is not a Tuple, then the
    returned algorithm is not type-stable).
  - `pType`: the problem type. Defaults to `:NLS` for `NonlinearProblem` and `:NLLS` for
    `NonlinearLeastSquaresProblem`. This is used to determine the correct problem type to
    dispatch on.

### Example

```julia
using NonlinearSolve

alg = NonlinearSolvePolyAlgorithm((NewtonRaphson(), Broyden()))
```
"""
struct NonlinearSolvePolyAlgorithm{pType, N, A} <: AbstractNonlinearSolveAlgorithm{:PolyAlg}
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
    for i in 1:N
        num = "  [$(i)]: "
        print(io, num)
        __show_algorithm(io, alg.algs[i], get_name(alg.algs[i]), length(num))
        i == N || println(io)
    end
end

@concrete mutable struct NonlinearSolvePolyAlgorithmCache{iip, N} <:
                         AbstractNonlinearSolveCache{iip}
    caches
    alg
    current::Int
end

function reinit_cache!(cache::NonlinearSolvePolyAlgorithmCache, args...; kwargs...)
    foreach(c -> reinit_cache!(c, args...; kwargs...), cache.caches)
    cache.current = 1
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
                            original = $(sol_syms[i]), trace = $(sol_syms[i]).trace)
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
                fus[idx]; retcode, stats, cache.caches[idx].trace)
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
                                original = $(cur_sol), trace = $(cur_sol).trace)
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
                                $(sol_syms[i]).stats, $(sol_syms[i]).trace)
                        end
                    end)
            end
            push!(calls, :(error("Current choices shouldn't get here!")))

            return Expr(:block, calls...)
        end
    end
end

"""
    RobustMultiNewton(::Type{T} = Float64; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, autodiff = nothing)

A polyalgorithm focused on robustness. It uses a mixture of Newton methods with different
globalizing techniques (trust region updates, line searches, etc.) in order to find a
method that is able to adequately solve the minimization problem.

Basically, if this algorithm fails, then "most" good ways of solving your problem fail and
you may need to think about reformulating the model (either there is an issue with the model,
or more precision / more stable linear solver choice is required).

### Arguments

  - `T`: The eltype of the initial guess. It is only used to check if some of the algorithms
    are compatible with the problem type. Defaults to `Float64`.
"""
function RobustMultiNewton(::Type{T} = Float64; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, autodiff = nothing) where {T}
    if __is_complex(T)
        # Let's atleast have something here for complex numbers
        algs = (NewtonRaphson(; concrete_jac, linsolve, precs, autodiff),)
    else
        algs = (TrustRegion(; concrete_jac, linsolve, precs),
            TrustRegion(; concrete_jac, linsolve, precs, autodiff,
                radius_update_scheme = RadiusUpdateSchemes.Bastin),
            NewtonRaphson(; concrete_jac, linsolve, precs,
                linesearch = LineSearchesJL(; method = BackTracking()), autodiff),
            TrustRegion(; concrete_jac, linsolve, precs,
                radius_update_scheme = RadiusUpdateSchemes.NLsolve, autodiff),
            TrustRegion(; concrete_jac, linsolve, precs,
                radius_update_scheme = RadiusUpdateSchemes.Fan, autodiff))
    end
    return NonlinearSolvePolyAlgorithm(algs, Val(:NLS))
end

"""
    FastShortcutNonlinearPolyalg(::Type{T} = Float64; concrete_jac = nothing,
        linsolve = nothing, precs = DEFAULT_PRECS, must_use_jacobian::Val = Val(false),
        prefer_simplenonlinearsolve::Val{SA} = Val(false), autodiff = nothing) where {T}

A polyalgorithm focused on balancing speed and robustness. It first tries less robust methods
for more performance and then tries more robust techniques if the faster ones fail.

### Arguments

  - `T`: The eltype of the initial guess. It is only used to check if some of the algorithms
    are compatible with the problem type. Defaults to `Float64`.
"""
function FastShortcutNonlinearPolyalg(::Type{T} = Float64; concrete_jac = nothing,
        linsolve = nothing, precs = DEFAULT_PRECS, must_use_jacobian::Val{JAC} = Val(false),
        prefer_simplenonlinearsolve::Val{SA} = Val(false),
        autodiff = nothing) where {T, JAC, SA}
    if JAC
        if __is_complex(T)
            algs = (NewtonRaphson(; concrete_jac, linsolve, precs, autodiff),)
        else
            algs = (NewtonRaphson(; concrete_jac, linsolve, precs, autodiff),
                NewtonRaphson(; concrete_jac, linsolve, precs,
                    linesearch = LineSearchesJL(; method = BackTracking()), autodiff),
                TrustRegion(; concrete_jac, linsolve, precs, autodiff),
                TrustRegion(; concrete_jac, linsolve, precs,
                    radius_update_scheme = RadiusUpdateSchemes.Bastin, autodiff))
        end
    else
        # SimpleNewtonRaphson and SimpleTrustRegion are not robust to singular Jacobians
        # and thus are not included in the polyalgorithm
        if SA
            if __is_complex(T)
                algs = (SimpleBroyden(),
                    Broyden(; init_jacobian = Val(:true_jacobian)),
                    SimpleKlement(),
                    NewtonRaphson(; concrete_jac, linsolve, precs, autodiff))
            else
                algs = (SimpleBroyden(),
                    Broyden(; init_jacobian = Val(:true_jacobian)),
                    SimpleKlement(),
                    NewtonRaphson(; concrete_jac, linsolve, precs, autodiff),
                    NewtonRaphson(; concrete_jac, linsolve, precs,
                        linesearch = LineSearchesJL(; method = BackTracking()), autodiff),
                    TrustRegion(; concrete_jac, linsolve, precs,
                        radius_update_scheme = RadiusUpdateSchemes.Bastin, autodiff))
            end
        else
            if __is_complex(T)
                algs = (Broyden(),
                    Broyden(; init_jacobian = Val(:true_jacobian)),
                    Klement(; linsolve, precs),
                    NewtonRaphson(; concrete_jac, linsolve, precs, autodiff))
            else
                algs = (Broyden(),
                    Broyden(; init_jacobian = Val(:true_jacobian)),
                    Klement(; linsolve, precs),
                    NewtonRaphson(; concrete_jac, linsolve, precs, autodiff),
                    NewtonRaphson(; concrete_jac, linsolve, precs,
                        linesearch = LineSearchesJL(; method = BackTracking()), autodiff),
                    TrustRegion(; concrete_jac, linsolve, precs, autodiff),
                    TrustRegion(; concrete_jac, linsolve, precs,
                        radius_update_scheme = RadiusUpdateSchemes.Bastin, autodiff))
            end
        end
    end
    return NonlinearSolvePolyAlgorithm(algs, Val(:NLS))
end

"""
    FastShortcutNLLSPolyalg(::Type{T} = Float64; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, kwargs...)

A polyalgorithm focused on balancing speed and robustness. It first tries less robust methods
for more performance and then tries more robust techniques if the faster ones fail.

### Arguments

  - `T`: The eltype of the initial guess. It is only used to check if some of the algorithms
    are compatible with the problem type. Defaults to `Float64`.
"""
function FastShortcutNLLSPolyalg(::Type{T} = Float64; concrete_jac = nothing,
        linsolve = nothing, precs = DEFAULT_PRECS, kwargs...) where {T}
    if __is_complex(T)
        algs = (GaussNewton(; concrete_jac, linsolve, precs, kwargs...),
            LevenbergMarquardt(; linsolve, precs, kwargs...))
    else
        algs = (GaussNewton(; concrete_jac, linsolve, precs, kwargs...),
            TrustRegion(; concrete_jac, linsolve, precs, kwargs...),
            GaussNewton(; concrete_jac, linsolve, precs,
                linesearch = LineSearchesJL(; method = BackTracking()), kwargs...),
            TrustRegion(; concrete_jac, linsolve, precs,
                radius_update_scheme = RadiusUpdateSchemes.Bastin, kwargs...),
            LevenbergMarquardt(; linsolve, precs, kwargs...))
    end
    return NonlinearSolvePolyAlgorithm(algs, Val(:NLLS))
end

## Defaults

## TODO: In the long run we want to use an `Assumptions` API like LinearSolve to specify
##       the conditioning of the Jacobian and such

## TODO: Currently some of the algorithms like LineSearches / TrustRegion don't support
##       complex numbers. We should use the `DiffEqBase` trait for this once all of the
##       NonlinearSolve algorithms support it. For now we just do a check and remove the
##       unsupported ones from default

## Defaults to a fast and robust poly algorithm in most cases. If the user went through
## the trouble of specifying a custom jacobian function, we should use algorithms that
## can use that!
function SciMLBase.__init(prob::NonlinearProblem, ::Nothing, args...; kwargs...)
    must_use_jacobian = Val(prob.f.jac !== nothing)
    return SciMLBase.__init(prob,
        FastShortcutNonlinearPolyalg(eltype(prob.u0); must_use_jacobian),
        args...; kwargs...)
end

function SciMLBase.__solve(prob::NonlinearProblem, ::Nothing, args...; kwargs...)
    must_use_jacobian = Val(prob.f.jac !== nothing)
    prefer_simplenonlinearsolve = Val(prob.u0 isa SArray)
    return SciMLBase.__solve(prob,
        FastShortcutNonlinearPolyalg(eltype(prob.u0); must_use_jacobian,
            prefer_simplenonlinearsolve), args...; kwargs...)
end

function SciMLBase.__init(prob::NonlinearLeastSquaresProblem, ::Nothing, args...; kwargs...)
    return SciMLBase.__init(prob, FastShortcutNLLSPolyalg(eltype(prob.u0)), args...;
        kwargs...)
end

function SciMLBase.__solve(prob::NonlinearLeastSquaresProblem, ::Nothing, args...;
        kwargs...)
    return SciMLBase.__solve(prob, FastShortcutNLLSPolyalg(eltype(prob.u0)), args...;
        kwargs...)
end
