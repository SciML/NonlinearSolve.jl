# Poly Algorithms
"""
    NonlinearSolvePolyAlgorithm(algs, ::Val{pType} = Val(:NLS);
        start_index = 1) where {pType}

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

### Keyword Arguments

  - `start_index`: the index to start at. Defaults to `1`.

### Example

```julia
using NonlinearSolve

alg = NonlinearSolvePolyAlgorithm((NewtonRaphson(), Broyden()))
```
"""
struct NonlinearSolvePolyAlgorithm{pType, N, A} <: AbstractNonlinearSolveAlgorithm{:PolyAlg}
    algs::A
    start_index::Int

    function NonlinearSolvePolyAlgorithm(
            algs, ::Val{pType} = Val(:NLS); start_index::Int = 1) where {pType}
        @assert pType ∈ (:NLS, :NLLS)
        @assert 0 < start_index ≤ length(algs)
        algs = Tuple(algs)
        return new{pType, length(algs), typeof(algs)}(algs, start_index)
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

@concrete mutable struct NonlinearSolvePolyAlgorithmCache{iip, N, timeit} <:
                         AbstractNonlinearSolveCache{iip, timeit}
    caches
    alg
    best::Int
    current::Int
    nsteps::Int
    stats::NLStats
    total_time::Float64
    maxtime
    retcode::ReturnCode.T
    force_stop::Bool
    maxiters::Int
    internalnorm
    u0
    u0_aliased
    alias_u0::Bool
end

function SymbolicIndexingInterface.symbolic_container(cache::NonlinearSolvePolyAlgorithmCache)
    cache.caches[cache.current]
end
SymbolicIndexingInterface.state_values(cache::NonlinearSolvePolyAlgorithmCache) = cache.u0

function Base.show(
        io::IO, cache::NonlinearSolvePolyAlgorithmCache{pType, N}) where {pType, N}
    problem_kind = ifelse(pType == :NLS, "NonlinearProblem", "NonlinearLeastSquaresProblem")
    println(io, "NonlinearSolvePolyAlgorithmCache for $(problem_kind) with $(N) algorithms")
    best_alg = ifelse(cache.best == -1, "nothing", cache.best)
    println(io, "Best algorithm: $(best_alg)")
    println(io, "Current algorithm: $(cache.current)")
    println(io, "nsteps: $(cache.nsteps)")
    println(io, "retcode: $(cache.retcode)")
    __show_cache(io, cache.caches[cache.current], 0)
end

function reinit_cache!(cache::NonlinearSolvePolyAlgorithmCache, args...; kwargs...)
    foreach(c -> reinit_cache!(c, args...; kwargs...), cache.caches)
    cache.current = cache.alg.start_index
    __reinit_internal!(cache.stats)
    cache.nsteps = 0
    cache.total_time = 0.0
end

for (probType, pType) in ((:NonlinearProblem, :NLS), (:NonlinearLeastSquaresProblem, :NLLS))
    algType = NonlinearSolvePolyAlgorithm{pType}
    @eval begin
        function SciMLBase.__init(
                prob::$probType, alg::$algType{N}, args...; stats = empty_nlstats(),
                maxtime = nothing, maxiters = 1000, internalnorm = DEFAULT_NORM,
                alias_u0 = false, verbose = true, kwargs...) where {N}
            if (alias_u0 && !ismutable(prob.u0))
                verbose && @warn "`alias_u0` has been set to `true`, but `u0` is \
                                  immutable (checked using `ArrayInterface.ismutable`)."
                alias_u0 = false  # If immutable don't care about aliasing
            end
            u0 = prob.u0
            if alias_u0
                u0_aliased = copy(u0)
            else
                u0_aliased = u0  # Irrelevant
            end
            alias_u0 && (prob = remake(prob; u0 = u0_aliased))
            return NonlinearSolvePolyAlgorithmCache{isinplace(prob), N, maxtime !== nothing}(
                map(
                    solver -> SciMLBase.__init(prob, solver, args...; stats, maxtime,
                        internalnorm, alias_u0, verbose, kwargs...),
                    alg.algs),
                alg,
                -1,
                alg.start_index,
                0,
                stats,
                0.0,
                maxtime,
                ReturnCode.Default,
                false,
                maxiters,
                internalnorm,
                u0,
                u0_aliased,
                alias_u0)
        end
    end
end

@generated function SciMLBase.solve!(cache::NonlinearSolvePolyAlgorithmCache{
        iip, N}) where {iip, N}
    calls = [quote
        1 ≤ cache.current ≤ length(cache.caches) ||
            error("Current choices shouldn't get here!")
    end]

    cache_syms = [gensym("cache") for i in 1:N]
    sol_syms = [gensym("sol") for i in 1:N]
    u_result_syms = [gensym("u_result") for i in 1:N]
    for i in 1:N
        push!(calls,
            quote
                $(cache_syms[i]) = cache.caches[$(i)]
                if $(i) == cache.current
                    cache.alias_u0 && copyto!(cache.u0_aliased, cache.u0)
                    $(sol_syms[i]) = SciMLBase.solve!($(cache_syms[i]))
                    if SciMLBase.successful_retcode($(sol_syms[i]))
                        stats = $(sol_syms[i]).stats
                        if cache.alias_u0
                            copyto!(cache.u0, $(sol_syms[i]).u)
                            $(u_result_syms[i]) = cache.u0
                        else
                            $(u_result_syms[i]) = $(sol_syms[i]).u
                        end
                        fu = get_fu($(cache_syms[i]))
                        return __build_solution_less_specialize(
                            $(sol_syms[i]).prob, cache.alg, $(u_result_syms[i]),
                            fu; retcode = $(sol_syms[i]).retcode, stats,
                            original = $(sol_syms[i]), trace = $(sol_syms[i]).trace)
                    elseif cache.alias_u0
                        # For safety we need to maintain a copy of the solution
                        $(u_result_syms[i]) = copy($(sol_syms[i]).u)
                    end
                    cache.current = $(i + 1)
                end
            end)
    end

    resids = map(x -> Symbol("$(x)_resid"), cache_syms)
    for (sym, resid) in zip(cache_syms, resids)
        push!(calls, :($(resid) = @isdefined($(sym)) ? get_fu($(sym)) : nothing))
    end
    push!(calls, quote
        fus = tuple($(Tuple(resids)...))
        minfu, idx = __findmin(cache.internalnorm, fus)
    end)
    for i in 1:N
        push!(calls, quote
            if idx == $(i)
                if cache.alias_u0
                    u = $(u_result_syms[i])
                else
                    u = get_u(cache.caches[$i])
                end
            end
        end)
    end
    push!(calls,
        quote
            retcode = cache.caches[idx].retcode
            if cache.alias_u0
                copyto!(cache.u0, u)
                u = cache.u0
            end
            return __build_solution_less_specialize(
                cache.caches[idx].prob, cache.alg, u, fus[idx];
                retcode, stats = cache.stats, cache.caches[idx].trace)
        end)

    return Expr(:block, calls...)
end

@generated function __step!(
        cache::NonlinearSolvePolyAlgorithmCache{iip, N}, args...; kwargs...) where {iip, N}
    calls = []
    cache_syms = [gensym("cache") for i in 1:N]
    for i in 1:N
        push!(calls,
            quote
                $(cache_syms[i]) = cache.caches[$(i)]
                if $(i) == cache.current
                    __step!($(cache_syms[i]), args...; kwargs...)
                    $(cache_syms[i]).nsteps += 1
                    if !not_terminated($(cache_syms[i]))
                        if SciMLBase.successful_retcode($(cache_syms[i]).retcode)
                            cache.best = $(i)
                            cache.force_stop = true
                            cache.retcode = $(cache_syms[i]).retcode
                        else
                            cache.current = $(i + 1)
                        end
                    end
                    return
                end
            end)
    end

    push!(calls, quote
        if !(1 ≤ cache.current ≤ length(cache.caches))
            minfu, idx = __findmin(cache.internalnorm, cache.caches)
            cache.best = idx
            cache.retcode = cache.caches[cache.best].retcode
            cache.force_stop = true
            return
        end
    end)

    return Expr(:block, calls...)
end

for (probType, pType) in ((:NonlinearProblem, :NLS), (:NonlinearLeastSquaresProblem, :NLLS))
    algType = NonlinearSolvePolyAlgorithm{pType}
    @eval begin
        @generated function SciMLBase.__solve(
                prob::$probType, alg::$algType{N}, args...; stats = empty_nlstats(),
                alias_u0 = false, verbose = true, kwargs...) where {N}
            sol_syms = [gensym("sol") for _ in 1:N]
            prob_syms = [gensym("prob") for _ in 1:N]
            u_result_syms = [gensym("u_result") for _ in 1:N]
            calls = [quote
                current = alg.start_index
                if (alias_u0 && !ismutable(prob.u0))
                    verbose && @warn "`alias_u0` has been set to `true`, but `u0` is \
                                      immutable (checked using `ArrayInterface.ismutable`)."
                    alias_u0 = false  # If immutable don't care about aliasing
                end
                u0 = prob.u0
                u0_aliased = alias_u0 ? zero(u0) : u0
            end]
            for i in 1:N
                cur_sol = sol_syms[i]
                push!(calls,
                    quote
                        if current == $i
                            if alias_u0
                                copyto!(u0_aliased, u0)
                                $(prob_syms[i]) = remake(prob; u0 = u0_aliased)
                            else
                                $(prob_syms[i]) = prob
                            end
                            $(cur_sol) = SciMLBase.__solve(
                                $(prob_syms[i]), alg.algs[$(i)], args...;
                                stats, alias_u0, verbose, kwargs...)
                            if SciMLBase.successful_retcode($(cur_sol))
                                if alias_u0
                                    copyto!(u0, $(cur_sol).u)
                                    $(u_result_syms[i]) = u0
                                else
                                    $(u_result_syms[i]) = $(cur_sol).u
                                end
                                return __build_solution_less_specialize(
                                    prob, alg, $(u_result_syms[i]), $(cur_sol).resid;
                                    $(cur_sol).retcode, $(cur_sol).stats,
                                    original = $(cur_sol), trace = $(cur_sol).trace)
                            elseif alias_u0
                                # For safety we need to maintain a copy of the solution
                                $(u_result_syms[i]) = copy($(cur_sol).u)
                            end
                            current = $(i + 1)
                        end
                    end)
            end

            resids = map(x -> Symbol("$(x)_resid"), sol_syms)
            for (sym, resid) in zip(sol_syms, resids)
                push!(calls, :($(resid) = @isdefined($(sym)) ? $(sym).resid : nothing))
            end

            push!(calls, quote
                resids = tuple($(Tuple(resids)...))
                minfu, idx = __findmin(DEFAULT_NORM, resids)
            end)

            for i in 1:N
                push!(calls,
                    quote
                        if idx == $i
                            if alias_u0
                                copyto!(u0, $(u_result_syms[i]))
                                $(u_result_syms[i]) = u0
                            else
                                $(u_result_syms[i]) = $(sol_syms[i]).u
                            end
                            return __build_solution_less_specialize(
                                prob, alg, $(u_result_syms[i]), $(sol_syms[i]).resid;
                                $(sol_syms[i]).retcode, $(sol_syms[i]).stats,
                                $(sol_syms[i]).trace, original = $(sol_syms[i]))
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
        algs = (TrustRegion(; concrete_jac, linsolve, precs, autodiff),
            TrustRegion(; concrete_jac, linsolve, precs, autodiff,
                radius_update_scheme = RadiusUpdateSchemes.Bastin),
            NewtonRaphson(; concrete_jac, linsolve, precs, autodiff),
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
        prefer_simplenonlinearsolve::Val{SA} = Val(false), autodiff = nothing,
        u0_len::Union{Int, Nothing} = nothing) where {T}

A polyalgorithm focused on balancing speed and robustness. It first tries less robust methods
for more performance and then tries more robust techniques if the faster ones fail.

### Arguments

  - `T`: The eltype of the initial guess. It is only used to check if some of the algorithms
    are compatible with the problem type. Defaults to `Float64`.

### Keyword Arguments

  - `u0_len`: The length of the initial guess. If this is `nothing`, then the length of the
    initial guess is not checked. If this is an integer and it is less than `25`, we use
    jacobian based methods.
"""
function FastShortcutNonlinearPolyalg(
        ::Type{T} = Float64; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, must_use_jacobian::Val{JAC} = Val(false),
        prefer_simplenonlinearsolve::Val{SA} = Val(false),
        u0_len::Union{Int, Nothing} = nothing, autodiff = nothing) where {T, JAC, SA}
    start_index = 1
    if JAC
        if __is_complex(T)
            algs = (NewtonRaphson(; concrete_jac, linsolve, precs, autodiff),)
        else
            algs = (NewtonRaphson(; concrete_jac, linsolve, precs, autodiff),
                NewtonRaphson(; concrete_jac, linsolve, precs,
                    linesearch = LineSearchesJL(; method = LineSearches.BackTracking()),
                    autodiff),
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
                    Broyden(; init_jacobian = Val(:true_jacobian), autodiff),
                    SimpleKlement(),
                    NewtonRaphson(; concrete_jac, linsolve, precs, autodiff))
            else
                start_index = u0_len !== nothing ? (u0_len ≤ 25 ? 4 : 1) : 1
                algs = (SimpleBroyden(),
                    Broyden(; init_jacobian = Val(:true_jacobian), autodiff),
                    SimpleKlement(),
                    NewtonRaphson(; concrete_jac, linsolve, precs, autodiff),
                    NewtonRaphson(; concrete_jac, linsolve, precs,
                        linesearch = LineSearchesJL(; method = LineSearches.BackTracking()),
                        autodiff),
                    TrustRegion(; concrete_jac, linsolve, precs,
                        radius_update_scheme = RadiusUpdateSchemes.Bastin, autodiff))
            end
        else
            if __is_complex(T)
                algs = (
                    Broyden(), Broyden(; init_jacobian = Val(:true_jacobian), autodiff),
                    Klement(; linsolve, precs, autodiff),
                    NewtonRaphson(; concrete_jac, linsolve, precs, autodiff))
            else
                # TODO: This number requires a bit rigorous testing
                start_index = u0_len !== nothing ? (u0_len ≤ 25 ? 4 : 1) : 1
                algs = (Broyden(; autodiff),
                    Broyden(; init_jacobian = Val(:true_jacobian), autodiff),
                    Klement(; linsolve, precs, autodiff),
                    NewtonRaphson(; concrete_jac, linsolve, precs, autodiff),
                    NewtonRaphson(; concrete_jac, linsolve, precs,
                        linesearch = LineSearchesJL(; method = LineSearches.BackTracking()),
                        autodiff),
                    TrustRegion(; concrete_jac, linsolve, precs, autodiff),
                    TrustRegion(; concrete_jac, linsolve, precs,
                        radius_update_scheme = RadiusUpdateSchemes.Bastin, autodiff))
            end
        end
    end
    return NonlinearSolvePolyAlgorithm(algs, Val(:NLS); start_index)
end

"""
    FastShortcutNLLSPolyalg(::Type{T} = Float64; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, autodiff = nothing, kwargs...)

A polyalgorithm focused on balancing speed and robustness. It first tries less robust methods
for more performance and then tries more robust techniques if the faster ones fail.

### Arguments

  - `T`: The eltype of the initial guess. It is only used to check if some of the algorithms
    are compatible with the problem type. Defaults to `Float64`.
"""
function FastShortcutNLLSPolyalg(
        ::Type{T} = Float64; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, autodiff = nothing, kwargs...) where {T}
    if __is_complex(T)
        algs = (GaussNewton(; concrete_jac, linsolve, precs, autodiff, kwargs...),
            LevenbergMarquardt(;
                linsolve, precs, autodiff, disable_geodesic = Val(true), kwargs...),
            LevenbergMarquardt(; linsolve, precs, autodiff, kwargs...))
    else
        algs = (GaussNewton(; concrete_jac, linsolve, precs, autodiff, kwargs...),
            LevenbergMarquardt(;
                linsolve, precs, disable_geodesic = Val(true), autodiff, kwargs...),
            TrustRegion(; concrete_jac, linsolve, precs, autodiff, kwargs...),
            GaussNewton(; concrete_jac, linsolve, precs,
                linesearch = LineSearchesJL(; method = BackTracking()),
                autodiff, kwargs...),
            TrustRegion(; concrete_jac, linsolve, precs,
                radius_update_scheme = RadiusUpdateSchemes.Bastin, autodiff, kwargs...),
            LevenbergMarquardt(; linsolve, precs, autodiff, kwargs...))
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
        FastShortcutNonlinearPolyalg(
            eltype(prob.u0); must_use_jacobian, u0_len = length(prob.u0)),
        args...;
        kwargs...)
end

function SciMLBase.__solve(prob::NonlinearProblem, ::Nothing, args...; kwargs...)
    must_use_jacobian = Val(prob.f.jac !== nothing)
    prefer_simplenonlinearsolve = Val(prob.u0 isa SArray)
    return SciMLBase.__solve(prob,
        FastShortcutNonlinearPolyalg(eltype(prob.u0); must_use_jacobian,
            prefer_simplenonlinearsolve, u0_len = length(prob.u0)),
        args...;
        kwargs...)
end

function SciMLBase.__init(prob::NonlinearLeastSquaresProblem, ::Nothing, args...; kwargs...)
    return SciMLBase.__init(
        prob, FastShortcutNLLSPolyalg(eltype(prob.u0)), args...; kwargs...)
end

function SciMLBase.__solve(
        prob::NonlinearLeastSquaresProblem, ::Nothing, args...; kwargs...)
    return SciMLBase.__solve(
        prob, FastShortcutNLLSPolyalg(eltype(prob.u0)), args...; kwargs...)
end
