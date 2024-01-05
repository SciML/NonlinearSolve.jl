# For spectral methods we currently only implement DF-SANE since after reading through
# papers, this seems to be the only one that is widely used. If we have a list of more
# papers we can see what is the right level of abstraction to implement here
@concrete struct GeneralizedDFSane{name} <: AbstractNonlinearSolveAlgorithm{name}
    linesearch
    σ_min
    σ_max
    σ_1
end

function Base.show(io::IO, alg::GeneralizedDFSane{name}) where {name}
    modifiers = String[]
    __is_present(alg.linesearch) && push!(modifiers, "linesearch = $(alg.linesearch)")
    push!(modifiers, "σ_min = $(alg.σ_min)")
    push!(modifiers, "σ_max = $(alg.σ_max)")
    push!(modifiers, "σ_1 = $(alg.σ_1)")
    print(io, "$(name)(\n    $(join(modifiers, ",\n    "))\n)")
end

concrete_jac(::GeneralizedDFSane) = nothing

@concrete mutable struct GeneralizedDFSaneCache{iip} <: AbstractNonlinearSolveCache{iip}
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
    nf::Int
    nsteps::Int
    maxiters::Int
    maxtime

    # Timer
    timer::TimerOutput
    total_time::Float64   # Simple Counter which works even if TimerOutput is disabled

    # Termination & Tracking
    termination_cache
    trace
    retcode::ReturnCode.T
    force_stop::Bool
end

@internal_caches GeneralizedDFSaneCache :linesearch_cache

function SciMLBase.__init(prob::AbstractNonlinearProblem, alg::GeneralizedDFSane, args...;
        alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
        termination_condition = nothing, internalnorm::F = DEFAULT_NORM, maxtime = Inf,
        kwargs...) where {F}
    timer = TimerOutput()
    @timeit_debug timer "cache construction" begin
        u = __maybe_unaliased(prob.u0, alias_u0)
        T = eltype(u)

        @bb du = similar(u)
        @bb u_cache = copy(u)
        fu = evaluate_f(prob, u)
        @bb fu_cache = copy(fu)

        linesearch_cache = init(prob, alg.linesearch, prob.f, fu, u, prob.p; maxiters,
            internalnorm, kwargs...)

        abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fu, u_cache,
            termination_condition)
        trace = init_nonlinearsolve_trace(alg, u, fu, nothing, du; kwargs...)

        return GeneralizedDFSaneCache{isinplace(prob)}(fu, fu_cache, u, u_cache, prob.p, du,
            alg, prob, T(alg.σ_1), T(alg.σ_min), T(alg.σ_max), linesearch_cache, 0, 0,
            maxiters, maxtime, timer, 0.0, tc_cache, trace, ReturnCode.Default, false)
    end
end

function __step!(cache::GeneralizedDFSaneCache{iip};
        recompute_jacobian::Union{Nothing, Bool} = nothing, kwargs...) where {iip}
    if recompute_jacobian !== nothing
        @warn "GeneralizedDFSane is a Jacobian-Free Algorithm. Ignoring \
              `recompute_jacobian`" maxlog=1
    end

    @timeit_debug cache.timer "descent" begin
        @bb @. cache.du = -cache.σ_n * cache.fu
    end

    @timeit_debug cache.timer "linesearch" begin
        linesearch_failed, α = solve!(cache.linesearch_cache, cache.u, cache.du)
    end

    if linesearch_failed
        cache.retcode = ReturnCode.InternalLineSearchFailed
        cache.force_stop = true
        return
    end

    @timeit_debug cache.timer "step" begin
        @bb axpy!(α, cache.du, cache.u)
        evaluate_f!(cache, cache.u, cache.p)
    end

    update_trace!(cache, α)
    check_and_update!(cache, cache.fu, cache.u, cache.u_cache)

    # Update Spectral Parameter
    @timeit_debug cache.timer "update spectral parameter" begin
        @bb @. cache.u_cache = cache.u - cache.u_cache
        @bb @. cache.fu_cache = cache.fu - cache.fu_cache

        cache.σ_n = dot(cache.u_cache, cache.u_cache) / dot(cache.u_cache, cache.fu_cache)

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
