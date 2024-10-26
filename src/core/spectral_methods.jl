# For spectral methods we currently only implement DF-SANE since after reading through
# papers, this seems to be the only one that is widely used. If we have a list of more
# papers we can see what is the right level of abstraction to implement here
"""
    GeneralizedDFSane{name}(linesearch, σ_min, σ_max, σ_1)

A generalized version of the DF-SANE algorithm. This algorithm is a Jacobian-Free Spectral
Method.

### Arguments

  - `linesearch`: Globalization using a Line Search Method. This is not optional currently,
    but that restriction might be lifted in the future.
  - `σ_min`: The minimum spectral parameter allowed. This is used to ensure that the
    spectral parameter is not too small.
  - `σ_max`: The maximum spectral parameter allowed. This is used to ensure that the
    spectral parameter is not too large.
  - `σ_1`: The initial spectral parameter. If this is not provided, then the algorithm
    initializes it as `σ_1 = <u, u> / <u, f(u)>`.
"""
@concrete struct GeneralizedDFSane{name} <: AbstractNonlinearSolveAlgorithm{name}
    linesearch
    σ_min
    σ_max
    σ_1
end

function __show_algorithm(io::IO, alg::GeneralizedDFSane, name, indent)
    modifiers = String[]
    __is_present(alg.linesearch) && push!(modifiers, "linesearch = $(alg.linesearch)")
    push!(modifiers, "σ_min = $(alg.σ_min)")
    push!(modifiers, "σ_max = $(alg.σ_max)")
    push!(modifiers, "σ_1 = $(alg.σ_1)")
    spacing = " "^indent * "    "
    spacing_last = " "^indent
    print(io, "$(name)(\n$(spacing)$(join(modifiers, ",\n$(spacing)"))\n$(spacing_last))")
end

# XXX: Remove
concrete_jac(::GeneralizedDFSane) = false

@concrete mutable struct GeneralizedDFSaneCache{iip, timeit} <:
                         AbstractNonlinearSolveCache{iip, timeit}
    # Basic Requirements
    fu
    fu_cache
    u
    u_cache
    p
    du
    alg
    prob

    # Parameters
    σ_n
    σ_min
    σ_max

    # Internal Caches
    linesearch_cache

    # Counters
    stats::NLStats
    nsteps::Int
    maxiters::Int
    maxtime

    # Timer
    timer
    total_time::Float64   # Simple Counter which works even if TimerOutput is disabled

    # Termination & Tracking
    termination_cache
    trace
    retcode::ReturnCode.T
    force_stop::Bool
    kwargs
end

function __reinit_internal!(
        cache::GeneralizedDFSaneCache{iip}, args...; p = cache.p, u0 = cache.u,
        alias_u0::Bool = false, maxiters = 1000, maxtime = nothing, kwargs...) where {iip}
    if iip
        recursivecopy!(cache.u, u0)
        cache.prob.f(cache.fu, cache.u, p)
    else
        cache.u = __maybe_unaliased(u0, alias_u0)
        set_fu!(cache, cache.prob.f(cache.u, p))
    end
    cache.p = p

    if cache.alg.σ_1 === nothing
        σ_n = dot(cache.u, cache.u) / dot(cache.u, cache.fu)
        # Spectral parameter bounds check
        if !(cache.alg.σ_min ≤ abs(σ_n) ≤ cache.alg.σ_max)
            test_norm = dot(cache.fu, cache.fu)
            σ_n = clamp(inv(test_norm), T(1), T(1e5))
        end
    else
        σ_n = T(cache.alg.σ_1)
    end
    cache.σ_n = σ_n

    reset_timer!(cache.timer)
    cache.total_time = 0.0

    reset!(cache.trace)
    reinit!(cache.termination_cache, get_fu(cache), get_u(cache); kwargs...)
    __reinit_internal!(cache.stats)
    cache.nsteps = 0
    cache.maxiters = maxiters
    cache.maxtime = maxtime
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
end

@internal_caches GeneralizedDFSaneCache :linesearch_cache

function SciMLBase.__init(prob::AbstractNonlinearProblem, alg::GeneralizedDFSane, args...;
        stats = empty_nlstats(), alias_u0 = false, maxiters = 1000,
        abstol = nothing, reltol = nothing, termination_condition = nothing,
        maxtime = nothing, kwargs...)
    timer = get_timer_output()
    @static_timeit timer "cache construction" begin
        u = __maybe_unaliased(prob.u0, alias_u0)
        T = eltype(u)

        @bb du = similar(u)
        @bb u_cache = copy(u)
        fu = evaluate_f(prob, u)
        @bb fu_cache = copy(fu)

        linesearch_cache = init(prob, alg.linesearch, fu, u; stats, kwargs...)

        abstol, reltol, tc_cache = NonlinearSolveBase.init_termination_cache(
            prob, abstol, reltol, fu, u_cache, termination_condition, Val(:regular))
        trace = init_nonlinearsolve_trace(prob, alg, u, fu, nothing, du; kwargs...)

        if alg.σ_1 === nothing
            σ_n = dot(u, u) / dot(u, fu)
            # Spectral parameter bounds check
            if !(alg.σ_min ≤ abs(σ_n) ≤ alg.σ_max)
                test_norm = dot(fu, fu)
                σ_n = clamp(inv(test_norm), T(1), T(1e5))
            end
        else
            σ_n = T(alg.σ_1)
        end

        return GeneralizedDFSaneCache{isinplace(prob), maxtime !== nothing}(
            fu, fu_cache, u, u_cache, prob.p, du, alg, prob, σ_n, T(alg.σ_min),
            T(alg.σ_max), linesearch_cache, stats, 0, maxiters, maxtime,
            timer, 0.0, tc_cache, trace, ReturnCode.Default, false, kwargs)
    end
end

function __step!(cache::GeneralizedDFSaneCache{iip};
        recompute_jacobian::Union{Nothing, Bool} = nothing, kwargs...) where {iip}
    if recompute_jacobian !== nothing
        @warn "GeneralizedDFSane is a Jacobian-Free Algorithm. Ignoring \
              `recompute_jacobian`" maxlog=1
    end

    @static_timeit cache.timer "descent" begin
        @bb @. cache.du = -cache.σ_n * cache.fu
    end

    @static_timeit cache.timer "linesearch" begin
        linesearch_sol = solve!(cache.linesearch_cache, cache.u, cache.du)
        linesearch_failed = !SciMLBase.successful_retcode(linesearch_sol.retcode)
        α = linesearch_sol.step_size
    end

    if linesearch_failed
        cache.retcode = ReturnCode.InternalLineSearchFailed
        cache.force_stop = true
        return
    end

    @static_timeit cache.timer "step" begin
        @bb axpy!(α, cache.du, cache.u)
        evaluate_f!(cache, cache.u, cache.p)
    end

    update_trace!(cache, α)
    check_and_update!(cache, cache.fu, cache.u, cache.u_cache)

    # Update Spectral Parameter
    @static_timeit cache.timer "update spectral parameter" begin
        @bb @. cache.u_cache = cache.u - cache.u_cache
        @bb @. cache.fu_cache = cache.fu - cache.fu_cache

        cache.σ_n = __dot(cache.u_cache, cache.u_cache) /
                    __dot(cache.u_cache, cache.fu_cache)

        # Spectral parameter bounds check
        if !(cache.σ_min ≤ abs(cache.σ_n) ≤ cache.σ_max)
            test_norm = dot(cache.fu, cache.fu)
            T = eltype(cache.σ_n)
            cache.σ_n = clamp(inv(test_norm), T(1), T(1e5))
        end
    end

    # Take step
    @bb copyto!(cache.u_cache, cache.u)
    @bb copyto!(cache.fu_cache, cache.fu)

    callback_into_cache!(cache, cache.linesearch_cache)

    return
end
