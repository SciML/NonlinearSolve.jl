#
# Driving a cache step by step, and reading a convergence rate off what it produced.
# Shared by the bar fixture and the Ferrite demonstration.
#
using MultiLevelNonlinearSolve
using LinearAlgebra

"""
    run_steps!(cache; maxsteps = 200, each = nothing, recompute_jacobian = nothing)

Drive `cache` to termination one `step!` at a time, calling `each(cache)` after every step.
Returns the number of steps taken.

`recompute_jacobian` is a predicate on the step index, for driving a frozen-Jacobian run.
Use `NonlinearSolveBase.solve_cache!` instead wherever nothing needs observing per step —
but note that it also runs the final termination-cache update, which this loop does not.
"""
function run_steps!(cache; maxsteps = 200, each = nothing, recompute_jacobian = nothing)
    k = 0
    while NonlinearSolveBase.not_terminated(cache) && k < maxsteps
        k += 1
        if recompute_jacobian === nothing
            step!(cache)
        else
            step!(cache; recompute_jacobian = recompute_jacobian(k))
        end
        each === nothing || each(cache)
    end
    return k
end

"""
    residual_history(cache, primary; chord_after = 0, maxsteps = 200)

`‖R̄‖_∞` per global iteration, starting from the initial residual. `chord_after` leading steps
recompute the Jacobian; later steps freeze it.
"""
function residual_history(
        cache, primary = cache.prob.f.primary; chord_after = 0, maxsteps = 200
    )
    residual(c) = norm(view(NonlinearSolveBase.get_fu(c), primary), Inf)
    e = [residual(cache)]
    run_steps!(
        cache; maxsteps, recompute_jacobian = chord_after == 0 ? nothing : ≤(chord_after),
        each = c -> push!(e, residual(c))
    )
    return e
end

"Observed order of each residual triple: `log(e_{k+1}/e_k) / log(e_k/e_{k-1})`."
observed_orders(e) = [log(e[k + 1] / e[k]) / log(e[k] / e[k - 1]) for k in 2:(length(e) - 1)]

"""
    tail_order(e; floor = 1e-14)

Observed order over the last triple whose three residuals are all above the roundoff floor.
Fitting the floored tail instead would report a linear rate no matter what the method does.
"""
function tail_order(e; floor = 1.0e-14)
    q = observed_orders(e)
    isempty(q) && return NaN
    clean = findall(k -> e[k + 2] > floor, 1:(length(e) - 2))
    return q[isempty(clean) ? length(q) : last(clean)]
end

"""
    is_superlinear(e; floor = 1e-14)

Whether the residual ratios `e_{k+1}/e_k` strictly decrease over the pre-floor segment.

This is the distinction that survives on a fixture converging in four iterations: a fitted
order needs an asymptotic window this problem never has, but a linear method (frozen
Jacobian) holds its ratio constant while a superlinear one keeps shrinking it.
"""
function is_superlinear(e; floor = 1.0e-14)
    keep = findall(>(floor), e)
    idx = first(keep):min(last(keep), length(e) - 1)
    length(idx) < 3 && return false
    r = [e[k + 1] / e[k] for k in idx]
    return all(r[k + 1] < r[k] for k in 1:(length(r) - 1))
end
