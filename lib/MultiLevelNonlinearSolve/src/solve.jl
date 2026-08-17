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

Four accuracy and preconditioning knobs act on this solve, and none of them is a field here:

  - *local forcing* — [`LocalToleranceSchedule`](@ref) on the function;
  - *linear forcing* — `forcing` on the `global_solver`;
  - *linear preconditioning* — `precs` on the `global_solver`'s `linsolve`;
  - *nonlinear preconditioning* — the `postcondition` solve keyword.

They are independent; the tutorial's taxonomy table says what each one buys.
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

# The state split and the commit callback are read straight off the problem's function rather
# than mirrored into the cache: `@concrete` types `prob`, so these stay type-stable, and there
# is only one place they can disagree.
@inline primary_range(cache::MultiLevelNewtonCache) = cache.prob.f.primary
@inline internal_range(cache::MultiLevelNewtonCache) = cache.prob.f.internal

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
    # With `alg = nothing` the predicate is exactly "this problem carries bounds".
    NonlinearSolveBase.needs_bounds_transform(prob, nothing) && throw(
        ArgumentError(
            "`MultiLevelNewton` does not support `lb`/`ub` bounds. Impose them inside the \
             condensed residual, or transform `ū` yourself."
        )
    )
    # `precondition` is refused by this function type's `compose_precondition` method, which
    # every public entry point reaches first — see `function.jl`.

    if haskey(kwargs, :alias_u0)
        alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = kwargs[:alias_u0])
    end
    verbose = NonlinearSolveBase.normalize_verbosity(verbose)
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
            local_tol = Base.RefValue{T}(initial_local_tolerance(schedule, local_tol_floor, T))
            p_condensed = LocalForcingParameters(prob.p, local_tol)
        end

        condensed_prob = SciMLBase.NonlinearProblem(
            wire_condensed_function(mlnf.f), u[primary], p_condensed
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
        # `u_cache` is the corrector's `u_prev` and is read nowhere else, so without one
        # configured it would only cost a full-state copy per iteration.
        u_cache = NonlinearSolveBase.get_postcondition(prob, kwargs) === nothing ?
            similar(u, 0) : copy(u)

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
            global_cache,
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

# `structdiff` rather than a generator: the generator's return type is opaque to inference,
# which makes the whole condensed `__init` below it dynamic.
without_postcondition(kwargs) =
    Base.structdiff(values(kwargs), NamedTuple{(:postcondition,)})

# ---------------------------------------------------------------------------------------
# step!
# ---------------------------------------------------------------------------------------

# How much worse the residual may get when the elimination is tightened before the solve is
# judged to have converged on an accuracy it could not actually support. Loose enough that
# ordinary rounding differences never trip it, tight enough to catch an elimination that was
# hiding a real error.
const TIGHTENING_DEMOTION_FACTOR = 10

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
    schedule.schedule === :fixed && return nothing
    exponent = schedule.schedule === :quadratic ? 2 : 1
    T = typeof(cache.local_tol[])
    residual = NonlinearSolveBase.L2_NORM(NonlinearSolveBase.get_fu(cache.global_cache))
    cache.local_tol[] = clamp(
        T(schedule.C) * T(residual)^exponent, cache.local_tol_floor, T(schedule.ceil)
    )
    return nothing
end

"""
    mirror_condensed!(cache)

Mirror the condensed cache's iterate and residual into the primary block of the full-length
state. The internal rows of `fu` are structurally zero — those equations were eliminated, not
solved — and are written once at `__init`/`reinit!`, never here: at FEM scale re-zeroing them
every iteration is a full pass over the internal state for no change.
"""
function mirror_condensed!(cache::MultiLevelNewtonCache)
    global_cache = cache.global_cache
    copyto!(view(cache.u, primary_range(cache)), NonlinearSolveBase.get_u(global_cache))
    copyto!(view(cache.fu, primary_range(cache)), NonlinearSolveBase.get_fu(global_cache))
    return cache
end

"""
    resync_condensed!(cache)

Re-evaluate `R̄` at whatever iterate the condensed cache currently holds and mirror it out.
Used wherever `ū` has moved behind the global solver's back — a corrector, a best-iterate
restore — or where `q` has changed under a fixed `ū`.
"""
function resync_condensed!(cache::MultiLevelNewtonCache)
    global_cache = cache.global_cache
    Utils.evaluate_f!(
        global_cache, NonlinearSolveBase.get_u(global_cache), global_cache.p
    )
    return mirror_condensed!(cache)
end

"The norm this solve's own termination condition judges the residual by."
function residual_norm(cache::MultiLevelNewtonCache, r)
    internalnorm = Utils.safe_getproperty(cache.termination_cache.mode, Val(:internalnorm))
    internalnorm === missing && return NonlinearSolveBase.Linf_NORM(r)
    return Utils.apply_norm(internalnorm, r)
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
    cache.u = NonlinearSolveBase.apply_postcondition!!(cache.u, cache.u_cache, cache)
    copyto!(
        NonlinearSolveBase.get_u(cache.global_cache), view(cache.u, primary_range(cache))
    )
    resync_condensed!(cache)
    return commit_local!(cache)
end

function commit_local!(cache::MultiLevelNewtonCache)
    cache.ncommits += 1
    return cache.prob.f.commit_internal!(
        view(cache.u, internal_range(cache)), NonlinearSolveBase.get_u(cache.global_cache),
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

    mirror_condensed!(cache)
    cache.local_ok = commit_local!(cache)
    NonlinearSolveBase.get_postcondition(cache) === nothing ||
        (cache.local_ok = apply_correction!(cache))

    copyto!(view(cache.du, primary_range(cache)), SciMLBase.get_du(global_cache))

    if cache.local_ok
        cache.retcode = global_cache.retcode
        cache.force_stop = global_cache.force_stop
    else
        cache.retcode = ReturnCode.ConvergenceFailure
        cache.force_stop = true
    end

    NonlinearSolveBase.update_trace!(cache, true)
    NonlinearSolveBase.get_postcondition(cache) === nothing ||
        copyto!(cache.u_cache, cache.u)

    # The regular commits use the tolerance derived from the *previous* residual, so the
    # committed `q` at the accepted root is one schedule step behind. Tighten it once, then
    # re-measure: the residual that passed the convergence test was computed through the
    # looser elimination, and a solve whose tightened residual misses `abstol` has not
    # reached a root it can support.
    if cache.force_stop && SciMLBase.successful_retcode(cache.retcode) &&
            cache.local_tol !== nothing
        accepted = residual_norm(cache, view(cache.fu, primary_range(cache)))
        scheduled = cache.local_tol[]
        cache.local_tol[] = cache.local_tol_floor
        cache.local_ok = commit_local!(cache)
        resync_condensed!(cache)
        cache.local_tol[] = scheduled
        tightened = residual_norm(cache, view(cache.fu, primary_range(cache)))
        if !cache.local_ok
            cache.retcode = ReturnCode.ConvergenceFailure
        elseif tightened > NonlinearSolveBase.get_abstol(cache) &&
                tightened > TIGHTENING_DEMOTION_FACTOR * accepted
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
        cache.local_tol[] = initial_local_tolerance(
            cache.prob.f.local_tolerance, cache.local_tol_floor, typeof(cache.local_tol[])
        )
        p_condensed = LocalForcingParameters(p, cache.local_tol)
    end

    # Both `u0` and `u` arrive at full length and must be sliced before they reach the
    # condensed cache, which iterates on `ū` alone. The slice goes into the condensed cache's
    # own iterate rather than a fresh vector, so restarting a time step allocates nothing —
    # and stays a `Vector`, which the linear cache's parameter type is pinned to.
    ū = NonlinearSolveBase.get_u(cache.global_cache)
    copyto!(ū, view(cache.u, primary_range(cache)))
    InternalAPI.reinit!(
        cache.global_cache, args...;
        p = p_condensed, u0 = ū, u = ū, maxiters, maxtime, kwargs...
    )

    # The condensed `reinit!` already evaluated `R̄` at the new start, so only mirror it.
    mirror_condensed!(cache)
    fill!(view(cache.fu, internal_range(cache)), zero(eltype(cache.fu)))
    cache.local_ok = commit_local!(cache)
    NonlinearSolveBase.get_postcondition(cache) === nothing ||
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
    # If this ever stops holding, the method below is no longer being selected and the
    # generic one is writing a condensed best-iterate into a full-length state.
    @assert length(tc_cache.u) == length(primary_range(cache))
    tc_cache.u == ū && return cache.fu
    copyto!(ū, tc_cache.u)
    resync_condensed!(cache)
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
