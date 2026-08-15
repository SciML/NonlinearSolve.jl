"""
    MultiLevelNewton(; global_solver = NewtonRaphson(), jacobian_reuse = :always,
                       name::Symbol = :MultiLevelNewton)

Multi-level Newton (Rabbat–Sangiovanni-Vincentelli–Hsieh) for a
[`MultiLevelNonlinearFunction`](@ref): eliminate the internal variables `q` at fixed `ū` with
one small local solve per point, then take a Newton step on the Schur-condensed system
`S·δū = -R̄`. The framework never sees the `ū`/`q` cross blocks — the user assembles `S` from
per-point correctors.

### Arguments

  - `global_solver`: the solver for the condensed problem over `ū`. It must support the
    iterator interface (`NewtonRaphson`, `TrustRegion`, … — anything that builds a stepping
    cache); the `SimpleNonlinearSolve` algorithms do not and are rejected at `init`.
  - `jacobian_reuse`: how often `S` is reassembled.

      + `:always` — every global iteration; the full multi-level Newton, quadratic.
      + `:chord` — assembled once, at the first iteration, then frozen; linear. Pair with a
        `:linear` [`LocalToleranceSchedule`](@ref).
      + a predicate `(cache) -> Bool` for anything in between.

    A `recompute_jacobian` keyword passed to `step!` overrides this for that step.

The four accuracy/preconditioning knobs are independent and never share a name: *local
forcing* is [`LocalToleranceSchedule`](@ref) on the function, *linear forcing* is `forcing`
on the `global_solver`, *linear preconditioning* is `precs` on the `global_solver`'s
`linsolve`, and *nonlinear preconditioning* is the `postcondition` solve keyword.
`MultiLevelNewton` itself exposes none of them.
"""
@concrete struct MultiLevelNewton <: AbstractNonlinearSolveAlgorithm
    global_solver
    jacobian_reuse
    name::Symbol
end

function MultiLevelNewton(;
        global_solver = NewtonRaphson(), jacobian_reuse = :always,
        name::Symbol = :MultiLevelNewton
    )
    if jacobian_reuse isa Symbol && jacobian_reuse ∉ (:always, :chord)
        throw(
            ArgumentError(
                "`jacobian_reuse` must be `:always`, `:chord` or a predicate on the cache; \
                 got `$(Meta.quot(jacobian_reuse))`."
            )
        )
    end
    return MultiLevelNewton(global_solver, jacobian_reuse, name)
end

# The corrector is applied on the full state at the commit point in `step!`, so support is
# unconditional — it does not depend on whether the global solver supports one, and the
# keyword is never forwarded into the condensed solve.
NonlinearSolveBase.supports_postcondition(::MultiLevelNewton) = true

@concrete mutable struct MultiLevelNewtonCache <: AbstractNonlinearSolveCache
    # Basic Requirements (full length: `[ū; q]`, with zero residual rows for `q`)
    fu
    u
    u_cache
    du
    p
    alg <: MultiLevelNewton
    prob <: AbstractNonlinearProblem

    # The condensed solve
    global_cache
    primary
    internal
    commit_internal!
    last_S

    # Local forcing
    local_tol      # `RefValue` owned by this cache, or `nothing`
    local_tol_floor
    local_ok::Bool

    # Counters
    stats::NLStats
    ncommits::Int
    nsteps::Int
    maxiters::Int
    maxtime

    # Timer
    timer
    total_time::Float64

    # Termination & Tracking
    termination_cache
    trace
    retcode::ReturnCode.T
    force_stop::Bool
    kwargs

    initializealg

    verbose
end

SciMLBase.get_du(cache::MultiLevelNewtonCache) = cache.du
NonlinearSolveBase.set_du!(cache::MultiLevelNewtonCache, δu) = (cache.du = δu)

"""
    ncommits(cache)

Number of commit steps a [`MultiLevelNewton`](@ref) cache has run, i.e. how often the local
ensemble was promoted to committed state. `cache.stats.nf` counts *condensed* residual
evaluations, each of which runs a full ensemble of trial local solves; neither counter can
see how many local iterations those took, which only the local solves themselves know.
"""
ncommits(cache::MultiLevelNewtonCache) = cache.ncommits

# ---------------------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------------------

function SciMLBase.__init(
        prob::AbstractNonlinearProblem, alg::MultiLevelNewton, args...;
        stats = NLStats(0, 0, 0, 0, 0),
        alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = false), maxiters = 1000,
        abstol = nothing, reltol = nothing, termination_condition = nothing,
        maxtime = nothing, verbose = NonlinearVerbosity(),
        initializealg = NonlinearSolveBase.NonlinearSolveDefaultInit(), kwargs...
    )
    mlnf = prob.f
    mlnf isa MultiLevelNonlinearFunction || throw(
        ArgumentError(
            "`MultiLevelNewton` needs the problem to carry a `MultiLevelNonlinearFunction`, \
             which is what tells it where `ū` and `q` live; got a `$(typeof(mlnf))`."
        )
    )
    # The bounds transform replaces `prob.f.f` with a full-space `BoundedWrapper`, and on
    # this problem that field *is* the condensed function the global solver is built from.
    if (hasfield(typeof(prob), :lb) && prob.lb !== nothing) ||
            (hasfield(typeof(prob), :ub) && prob.ub !== nothing)
        throw(
            ArgumentError(
                "`MultiLevelNewton` does not support `lb`/`ub` bounds. Impose them inside \
                 the condensed residual, or transform `ū` yourself."
            )
        )
    end
    haskey(kwargs, :precondition) && throw(
        ArgumentError(
            "`precondition` is not supported on a multi-level problem: the corrector `G` \
             acts on the full residual, which has length `n`, while the solved system is \
             the condensed one of length `n̄`. Compose `G` into your own condensed residual \
             instead."
        )
    )

    if haskey(kwargs, :alias_u0)
        alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = kwargs[:alias_u0])
    end
    verbose = normalize_verbosity(verbose)
    timer = get_timer_output()

    @static_timeit timer "cache construction" begin
        u = Utils.maybe_unaliased(prob.u0, alias.alias_u0)
        T = eltype(u)
        primary, internal = mlnf.primary, mlnf.internal

        # One tolerance cell per cache, not per problem: two concurrent solves of the same
        # problem object must not read each other's local tolerance, or a trial residual
        # stops being reproducible. Seeded here, before the condensed `__init` below, which
        # evaluates `R̄(ū₀)`.
        schedule = mlnf.local_tolerance
        local_tol_floor = zero(T)
        local_tol = nothing
        p_condensed = prob.p
        if schedule !== nothing
            local_tol_floor = T(schedule.floor_rel) *
                NonlinearSolveBase.get_tolerance(abstol, T)
            local_tol = Base.RefValue{T}(max(T(schedule.tol_init), local_tol_floor))
            p_condensed = LocalForcingParameters(prob.p, local_tol)
        end

        last_S = Base.RefValue{Any}(nothing)
        condensed_prob = SciMLBase.NonlinearProblem(
            wire_condensed_function(mlnf.f, last_S), u[primary], p_condensed
        )

        # `:stats` is not an allowed keyword of the public `init`, so the condensed cache is
        # built through `SciMLBase.__init` in order to share one `NLStats` object rather
        # than copying counters between caches (polyalg.jl:285 does the same).
        # `:postcondition` is stripped: the user's corrector is written for the full state.
        global_cache = SciMLBase.__init(
            condensed_prob, alg.global_solver, args...;
            stats, maxiters, abstol, reltol, termination_condition, maxtime, verbose,
            initializealg = SciMLBase.NoInit(), without_postcondition(kwargs)...
        )
        applicable(InternalAPI.step!, global_cache) || throw(
            ArgumentError(
                "`global_solver = $(alg.global_solver)` does not support the nonlinear \
                 solver iterator interface, which `MultiLevelNewton` drives one step at a \
                 time. Use an algorithm that builds a stepping cache, e.g. \
                 `NewtonRaphson()` or `TrustRegion()`."
            )
        )
        termination_cache = Utils.safe_getproperty(global_cache, Val(:termination_cache))
        termination_cache === missing && throw(
            ArgumentError(
                "`global_solver = $(alg.global_solver)` builds a cache without a \
                 `termination_cache`, which `MultiLevelNewton` needs in order to report \
                 tolerances and the best iterate."
            )
        )

        fu = Utils.safe_similar(u)
        fill!(fu, zero(T))
        copyto!(view(fu, primary), NonlinearSolveBase.get_fu(global_cache))
        du = Utils.safe_similar(u)
        fill!(du, zero(T))
        u_cache = copy(u)

        # Commit once here so `q` is consistent with `ū₀` before the first trial warm-starts
        # from it, and so `sol.u` is meaningful even for a zero-iteration solve.
        local_ok = mlnf.commit_internal!(
            view(u, internal), NonlinearSolveBase.get_u(global_cache), p_condensed
        )

        trace = NonlinearSolveBase.init_nonlinearsolve_trace(
            prob, alg, u, fu, nothing, du; kwargs...
        )

        cache = MultiLevelNewtonCache(
            fu, u, u_cache, du, prob.p, alg, prob,
            global_cache, primary, internal, mlnf.commit_internal!, last_S,
            local_tol, local_tol_floor, local_ok,
            stats, 1, 0, maxiters, maxtime,
            timer, 0.0,
            termination_cache, trace, ReturnCode.Default, false, kwargs, initializealg,
            verbose
        )
        NonlinearSolveBase.run_initialization!(cache)
    end

    return cache
end

function normalize_verbosity(verbose)
    verbose isa Bool && return verbose ? NonlinearVerbosity() : NonlinearVerbosity(None())
    verbose isa AbstractVerbosityPreset && return NonlinearVerbosity(verbose)
    return verbose
end

without_postcondition(kwargs) =
    (; (k => v for (k, v) in pairs(kwargs) if k !== :postcondition)...)

# ---------------------------------------------------------------------------------------
# step!
# ---------------------------------------------------------------------------------------

function reuse_decision(cache::MultiLevelNewtonCache)
    reuse = cache.alg.jacobian_reuse
    reuse === :always && return true
    # `:chord` still has to assemble once: at the first step the Jacobian storage holds
    # nothing but whatever `similar(jac_prototype)` left there.
    reuse === :chord && return cache.nsteps == 0
    return reuse(cache)::Bool
end

function update_local_tolerance!(cache::MultiLevelNewtonCache)
    cache.local_tol === nothing && return nothing
    schedule = cache.prob.f.local_tolerance
    exponent = local_forcing_exponent(schedule)
    exponent == 0 && return nothing
    T = typeof(cache.local_tol[])
    residual = NonlinearSolveBase.L2_NORM(NonlinearSolveBase.get_fu(cache.global_cache))
    cache.local_tol[] = clamp(
        T(schedule.C) * T(residual)^exponent, cache.local_tol_floor, T(schedule.ceil)
    )
    return nothing
end

"""
    apply_correction!(cache) -> local_ok

Apply the user's `postcondition` corrector to the *full* state and restore consistency
around it: the corrected `ū` goes back into the condensed cache, `R̄` is re-evaluated there,
and `q` is re-committed at the corrected iterate.

The corrector runs here, at the commit point, rather than inside the condensed solve, which
only ever sees a length-`n̄` vector the corrector was not written for.
"""
function apply_correction!(cache::MultiLevelNewtonCache)
    global_cache = cache.global_cache
    cache.u = NonlinearSolveBase.apply_postcondition!!(cache.u, cache.u_cache, cache)
    ū = NonlinearSolveBase.get_u(global_cache)
    copyto!(ū, view(cache.u, cache.primary))
    Utils.evaluate_f!(global_cache, ū, global_cache.p)
    copyto!(view(cache.fu, cache.primary), NonlinearSolveBase.get_fu(global_cache))
    return commit_local!(cache)
end

function commit_local!(cache::MultiLevelNewtonCache)
    cache.ncommits += 1
    return cache.commit_internal!(
        view(cache.u, cache.internal), NonlinearSolveBase.get_u(cache.global_cache),
        cache.global_cache.p
    )
end

function InternalAPI.step!(
        cache::MultiLevelNewtonCache; recompute_jacobian::Union{Nothing, Bool} = nothing
    )
    global_cache = cache.global_cache

    update_local_tolerance!(cache)

    rj = recompute_jacobian === nothing ? reuse_decision(cache) : recompute_jacobian
    @static_timeit cache.timer "global step" begin
        InternalAPI.step!(global_cache; recompute_jacobian = rj)
    end
    # `InternalAPI.step!` bypasses `CommonSolve.step!`, which is the only place `nsteps` is
    # incremented (NSB/solve.jl:809). Left at 0 the global solver's Eisenstat–Walker forcing
    # resets η to η₀ on every iteration instead of adapting, and every trace entry is
    # labelled iteration 1. polyalg.jl:311-312 does this for the same reason.
    global_cache.nsteps += 1

    copyto!(view(cache.u, cache.primary), NonlinearSolveBase.get_u(global_cache))
    copyto!(view(cache.fu, cache.primary), NonlinearSolveBase.get_fu(global_cache))
    cache.local_ok = commit_local!(cache)
    NonlinearSolveBase.get_postcondition(cache) === nothing ||
        (cache.local_ok = apply_correction!(cache))

    copyto!(view(cache.du, cache.primary), SciMLBase.get_du(global_cache))

    if cache.local_ok
        cache.retcode = global_cache.retcode
        cache.force_stop = global_cache.force_stop
    else
        cache.retcode = ReturnCode.ConvergenceFailure
        cache.force_stop = true
    end

    NonlinearSolveBase.update_trace!(cache, true)
    copyto!(cache.u_cache, cache.u)

    # The regular commits use the tolerance derived from the *previous* residual, so the
    # committed `q` at the accepted root is one schedule step behind. Tighten it once, then
    # re-measure: the residual that passed the convergence test was computed through the
    # looser elimination, and a solve whose tightened residual misses `abstol` has not
    # reached a root it can support.
    if cache.force_stop && SciMLBase.successful_retcode(cache.retcode) &&
            cache.local_tol !== nothing
        accepted = maximum(abs, view(cache.fu, cache.primary))
        scheduled = cache.local_tol[]
        cache.local_tol[] = cache.local_tol_floor
        cache.local_ok = commit_local!(cache)
        ū = NonlinearSolveBase.get_u(global_cache)
        Utils.evaluate_f!(global_cache, ū, global_cache.p)
        copyto!(view(cache.fu, cache.primary), NonlinearSolveBase.get_fu(global_cache))
        cache.local_tol[] = scheduled
        tightened = maximum(abs, view(cache.fu, cache.primary))
        if !cache.local_ok
            cache.retcode = ReturnCode.ConvergenceFailure
        elseif tightened > NonlinearSolveBase.get_abstol(cache) &&
                tightened > 10 * accepted
            # Both conditions are needed. The ratio alone is noisy once the residual sits at
            # the roundoff floor; `abstol` alone would misfire whenever termination is driven
            # by `reltol` on a large-scale residual.
            cache.retcode = ReturnCode.Stalled
        end
    end

    return nothing
end

# ---------------------------------------------------------------------------------------
# reinit! — hand-written on purpose
# ---------------------------------------------------------------------------------------

# `@internal_caches` would forward the *full-length* `u0`/`u` into the condensed cache's
# `reinit_common!`, which `recursivecopy!`s them into an `n̄`-length buffer
# (`DimensionMismatch`) and then runs the full-length residual, i.e. a whole extra ensemble
# trial. Slice first instead.
function InternalAPI.reinit!(
        cache::MultiLevelNewtonCache, args...; p = cache.p, u0 = cache.u, u = cache.u,
        maxiters = cache.maxiters, maxtime = cache.maxtime, kwargs...
    )
    cache.p = p
    u0 === cache.u || copyto!(cache.u, u0)

    p_condensed = p
    if cache.local_tol !== nothing
        cache.local_tol[] = max(
            typeof(cache.local_tol[])(cache.prob.f.local_tolerance.tol_init),
            cache.local_tol_floor
        )
        p_condensed = LocalForcingParameters(p, cache.local_tol)
    end

    # Both `u0` and `u` arrive at full length and must be sliced before they reach the
    # condensed cache, which iterates on `ū` alone.
    ū = cache.u[cache.primary]
    InternalAPI.reinit!(
        cache.global_cache, args...;
        p = p_condensed, u0 = ū, u = ū, maxiters, maxtime, kwargs...
    )

    copyto!(view(cache.fu, cache.primary), NonlinearSolveBase.get_fu(cache.global_cache))
    fill!(view(cache.fu, cache.internal), zero(eltype(cache.fu)))
    cache.local_ok = commit_local!(cache)
    copyto!(cache.u_cache, cache.u)

    NonlinearSolveBase.reset_timer!(cache.timer)
    cache.total_time = 0.0
    NonlinearSolveBase.reset!(cache.trace)
    cache.ncommits = 1
    cache.nsteps = 0
    cache.maxiters = maxiters
    cache.maxtime = maxtime
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
    return cache
end

# ---------------------------------------------------------------------------------------
# termination-cache overrides
# ---------------------------------------------------------------------------------------

# Both branches are reached with the *condensed* termination cache (it is aliased into this
# cache), so the generic methods would work in `ū` coordinates on a full-length state.
#
# Safe-best is the default mode (`AbsNormSafeBestTerminationMode`) and therefore the hot
# path: it restores the best-so-far iterate, which here is a `ū` of length `n̄` that the
# generic method would `copyto!` straight into the full `u`. Restoring `ū` also invalidates
# `q`, so the ensemble is re-run and re-committed at the restored iterate.
function NonlinearSolveBase.update_from_termination_cache!(
        tc_cache, cache::MultiLevelNewtonCache,
        ::NonlinearSolveBase.AbstractSafeBestNonlinearTerminationMode,
        u = NonlinearSolveBase.get_u(cache)
    )
    global_cache = cache.global_cache
    ū = NonlinearSolveBase.get_u(global_cache)
    # When the global solver terminates from inside a step it has already restored its own
    # best iterate, so `ū` is that iterate and `step!` synced everything from it. Re-running
    # the ensemble then would cost a whole extra elimination per solve. The restore below is
    # for the other exits — a failed commit, `maxiters` — where nothing restored it.
    tc_cache.u == ū && return cache.fu
    copyto!(ū, tc_cache.u)
    Utils.evaluate_f!(global_cache, ū, global_cache.p)
    copyto!(view(cache.u, cache.primary), ū)
    copyto!(view(cache.fu, cache.primary), NonlinearSolveBase.get_fu(global_cache))
    cache.local_ok = commit_local!(cache)
    # The restored iterate comes from the global solver's own history, which records states
    # from *before* the corrector ran. Re-apply it, or the returned state would violate a
    # corrector every accepted iterate satisfied.
    NonlinearSolveBase.get_postcondition(cache) === nothing ||
        (cache.local_ok = apply_correction!(cache))
    return cache.fu
end

# The plain branch would re-evaluate the *full-length* residual, i.e. run one more complete
# ensemble trial at an iterate whose `q` is already committed. `step!` left `u`/`fu` in sync
# with the condensed cache, so there is nothing left to do.
function NonlinearSolveBase.update_from_termination_cache!(
        tc_cache, cache::MultiLevelNewtonCache,
        ::NonlinearSolveBase.AbstractNonlinearTerminationMode,
        u = NonlinearSolveBase.get_u(cache)
    )
    return cache.fu
end
