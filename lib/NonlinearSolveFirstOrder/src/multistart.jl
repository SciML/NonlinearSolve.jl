"""
    SobolMultistart(alg = FastShortcutBoundedPolyalg(); nstarts::Int = 16,
        search_scale = 10, early_exit::Bool = true, restoration::Bool = true,
        restoration_alg = nothing)

A deterministic multistart wrapper for `NonlinearProblem` and
`NonlinearLeastSquaresProblem`, intended for box-constrained solves where a
local method can converge to a constrained stationary point on an active bound
that is not a root of the system. This wrapper is the default algorithm for
problems with `lb` or `ub`; select `alg` directly for a purely local solve
without restarts.

The wrapper runs `alg` from `nstarts` deterministic start points and returns
the best result. The first start is always `prob.u0`, so the wrapper is never
worse than a direct local solve; the remaining starts are points of a Sobol
low-discrepancy sequence over the search region. Every start goes through the
ordinary `solve`/`init` path, so `linsolve`, `concrete_jac`, sparse
`jac_prototype`, JVP/VJP callbacks, AD backends, and keyword arguments such as
`abstol`, `reltol`, and `maxiters` apply to each sub-solve exactly as they
would to a direct `solve(prob, alg; kwargs...)` call.

### Arguments

  - `alg`: the local algorithm run from each start. Defaults to
    [`FastShortcutBoundedPolyalg`](@ref).

### Keyword Arguments

  - `nstarts`: the number of start points, including `prob.u0`. Defaults to 16.
  - `search_scale`: components with an infinite bound are sampled over a window
    of half-width `search_scale * max(1, |u0_i|)` centered on `u0_i`; components
    with a finite bound use the bound itself as the edge of the sampling
    region. Defaults to 10.
  - `early_exit`: return immediately once a sub-solve reports a successful
    retcode. With `early_exit = false` every start runs and the feasible
    solution with the lowest residual norm is returned. Defaults to `true`.
  - `restoration`: when a start stalls on an active bound, run one unbounded
    restoration probe from the stalled iterate with the nonfixed bounds
    relaxed. A probe that converges inside the original box returns that root
    as the start's result; a root outside the box confirms the boundary point
    is a genuine constrained solution and the stalled result is kept.
    Defaults to `true`.
  - `restoration_alg`: the algorithm used for restoration probes. Defaults to
    `alg`.

The returned `retcode` is the sub-solve's own retcode on success and
`ReturnCode.Stalled` when no start finds a solution; a failure is never
reported as a success. The winning sub-solve is stored in the solution's
`original` field. The wrapper also applies to problems without bounds, where
all starts are drawn from the `search_scale` windows around `u0`.
"""
@concrete struct SobolMultistart <: AbstractNonlinearSolveAlgorithm
    alg
    restoration_alg
    nstarts::Int
    search_scale
    early_exit::Bool
    restoration::Bool
end

function SobolMultistart(
        alg = FastShortcutBoundedPolyalg();
        nstarts::Int = 16, search_scale = 10, early_exit::Bool = true,
        restoration::Bool = true, restoration_alg = nothing
    )
    nstarts >= 1 ||
        throw(ArgumentError("`nstarts` must be at least 1."))
    search_scale > 0 ||
        throw(ArgumentError("`search_scale` must be positive."))
    return SobolMultistart(
        alg, restoration_alg === nothing ? alg : restoration_alg,
        nstarts, search_scale, early_exit, restoration
    )
end

SciMLBase.allowsbounds(::SobolMultistart) = true

function NonlinearSolveBase.supports_postcondition(alg::SobolMultistart)
    return NonlinearSolveBase.supports_postcondition(alg.alg)
end

function NonlinearSolveBase.prepare_default_bounds(prob, ::SobolMultistart)
    return NonlinearSolveBase.prepare_default_bounds(prob, nothing)
end

function _multistart_search_region(prob, scale)
    u0 = prob.u0
    x = _box_vector(u0)
    lb, ub = _box_bounds(prob, u0)
    T = eltype(x)
    halfwidth = T(scale) .* max.(one(T), abs.(x))
    lo = map((l, xi, w) -> isfinite(l) ? l : xi - w, lb, x, halfwidth)
    hi = map((h, xi, w) -> isfinite(h) ? h : xi + w, ub, x, halfwidth)
    return x, lb, ub, min.(lo, hi), max.(lo, hi)
end

function _multistart_starts(prob, nstarts, scale)
    x0, lb, ub, lo, hi = _multistart_search_region(prob, scale)
    T = eltype(x0)
    starts = Vector{Vector{T}}(undef, nstarts)
    starts[1] = T.(clamp.(x0, lb, ub))
    seq = Sobol.SobolSeq(lo, hi)
    for i in 2:nstarts
        starts[i] = T.(Sobol.next!(seq))
    end
    return starts
end

_multistart_fatal(err) =
    err isa InterruptException || err isa OutOfMemoryError || err isa StackOverflowError

function _multistart_feasible(x, lb, ub)
    T = eltype(x)
    tol = eps(T)^(3 // 4)
    for i in eachindex(x)
        slack = tol * max(one(T), abs(lb[i]), abs(ub[i]), abs(x[i]))
        (x[i] < lb[i] - slack || x[i] > ub[i] + slack) && return false
    end
    return true
end

function _multistart_on_active_bound(x, lb, ub)
    T = eltype(x)
    tol = eps(T)^(3 // 4)
    for i in eachindex(x)
        lb[i] == ub[i] && continue
        slack = tol * max(one(T), abs(lb[i]), abs(ub[i]), abs(x[i]))
        (isfinite(lb[i]) && x[i] - lb[i] <= slack) && return true
        (isfinite(ub[i]) && ub[i] - x[i] <= slack) && return true
    end
    return false
end

function _multistart_restoration(prob, u_stalled, alg, args...; kwargs...)
    x = _box_vector(u_stalled)
    lb, ub = _box_bounds(prob, u_stalled)
    _multistart_on_active_bound(x, lb, ub) || return nothing
    probe = if all(l != h for (l, h) in zip(lb, ub))
        SciMLBase.remake(prob; u0 = u_stalled, lb = nothing, ub = nothing)
    else
        new_lb = map((l, h) -> l == h ? l : -Inf, lb, ub)
        new_ub = map((l, h) -> l == h ? h : Inf, lb, ub)
        SciMLBase.remake(prob; u0 = u_stalled, lb = new_lb, ub = new_ub)
    end
    return try
        SciMLBase.__solve(probe, alg, args...; kwargs...)
    catch err
        _multistart_fatal(err) && rethrow()
        nothing
    end
end

function _multistart_build_solution(prob, alg, sol, stats; retcode = sol.retcode)
    return SciMLBase.build_solution(
        prob, alg, sol.u, sol.resid; retcode, original = sol, stats
    )
end

function SciMLBase.__solve(
        prob::AbstractNonlinearProblem, alg::SobolMultistart, args...; kwargs...
    )
    # Initialization callbacks run once for the whole solve, not per start.
    init_alg = get(
        kwargs, :initializealg, NonlinearSolveBase.NonlinearSolveDefaultInit()
    )
    prob, init_success = NonlinearSolveBase.run_initialization!(prob, init_alg, prob)
    if !init_success
        u = _box_state(prob.u0, _box_vector(prob.u0))
        return SciMLBase.build_solution(
            prob, alg, u, Utils.evaluate_f(prob, u);
            retcode = ReturnCode.InitialFailure
        )
    end
    sub_kwargs = merge((; kwargs...), (; initializealg = SciMLBase.NoInit()))

    starts = _multistart_starts(prob, alg.nstarts, alg.search_scale)
    _, lb, ub, _, _ = _multistart_search_region(prob, alg.search_scale)
    internalnorm = get(kwargs, :internalnorm, L2_NORM)
    stats = NLStats(0, 0, 0, 0, 0)
    best = nothing
    best_norm = Inf
    first_error = nothing
    warned = false

    for x in starts
        sub_prob = SciMLBase.remake(prob; u0 = _box_state(prob.u0, x))
        sol = try
            SciMLBase.__solve(sub_prob, alg.alg, args...; sub_kwargs...)
        catch err
            _multistart_fatal(err) && rethrow()
            first_error === nothing && (first_error = err)
            continue
        end
        sol.stats !== nothing && (stats = Base.merge(stats, sol.stats))

        if alg.restoration && sol.retcode === ReturnCode.Stalled &&
                sol.resid !== nothing && !iszero(internalnorm(sol.resid))
            probe = _multistart_restoration(
                prob, sol.u, alg.restoration_alg, args...; sub_kwargs...
            )
            if probe !== nothing
                probe.stats !== nothing && (stats = Base.merge(stats, probe.stats))
                if SciMLBase.successful_retcode(probe.retcode)
                    if _multistart_feasible(_box_vector(probe.u), lb, ub)
                        sol = probe
                    elseif !warned
                        @warn "SobolMultistart restoration converged to a root \
                            outside the bounds at u = $(probe.u); the stalled \
                            point is a genuine constrained solution."
                        warned = true
                    end
                end
            end
        end

        if alg.early_exit && SciMLBase.successful_retcode(sol.retcode)
            return _multistart_build_solution(prob, alg, sol, stats)
        end

        resid_norm = _multistart_feasible(_box_vector(sol.u), lb, ub) ?
            internalnorm(sol.resid) : Inf
        if resid_norm < best_norm
            best, best_norm = sol, resid_norm
        end
    end

    if best === nothing
        first_error !== nothing && throw(first_error)
        u = _box_state(prob.u0, clamp.(_box_vector(prob.u0), lb, ub))
        resid = Utils.evaluate_f(prob, u)
        return SciMLBase.build_solution(
            prob, alg, u, resid; retcode = ReturnCode.Stalled, stats
        )
    end
    SciMLBase.successful_retcode(best.retcode) &&
        return _multistart_build_solution(prob, alg, best, stats)
    return _multistart_build_solution(prob, alg, best, stats; retcode = ReturnCode.Stalled)
end
