"""
    NewtonBisection()

Newton's method safeguarded by bisection, for problems that give the derivative of `f` as
the `jac` of their `IntervalNonlinearFunction`:

```julia
f(u, p) = u^2 - p
df(u, p) = 2u
prob = IntervalNonlinearProblem(IntervalNonlinearFunction(f; jac = df), (1.0, 2.0), 2.0)
sol = solve(prob, NewtonBisection())
```

Newton steps start from the end of the bracket with the smaller residual. The method bisects
instead when the step leaves the bracket or is not shorter than half the step before the
previous one, as `rtsafe` of Numerical Recipes does. When successive Newton steps from the
same end shrink by a ratio `ρ > 1/2`, as near a root of multiplicity `m ≈ 1 / (1 - ρ)`, the
step is multiplied by that `m`. Once the step is within the tolerance, the next point is
placed just beyond the Newton point, on the other side of the root, so that both ends of the
bracket converge.

Near a simple root the convergence is quadratic. Each iteration evaluates `f` once, and
`jac` only at the start of each Newton step. The method pays off when the derivative is
cheap compared with `f`.
"""
struct NewtonBisection <: AbstractBracketingAlgorithm end

function SciMLBase.__solve(
        prob::IntervalNonlinearProblem, alg::NewtonBisection, args...;
        maxiters = 1000, abstol = nothing, verbose::NonlinearVerbosity = NonlinearVerbosity(), kwargs...
    )
    @assert !SciMLBase.isinplace(prob) "`NewtonBisection` only supports out-of-place problems."

    left, right = minmax(promote(prob.tspan...)...)
    fun = interval_derivative(prob, alg)
    fl, fr = value(fun, left), value(fun, right)

    abstol = NonlinearSolveBase.get_tolerance(
        left, abstol, promote_type(eltype(left), eltype(right))
    )

    if iszero(fl)
        return build_exact_solution(prob, alg, left, fl, ReturnCode.ExactSolutionLeft)
    end

    if iszero(fr)
        return build_exact_solution(prob, alg, right, fr, ReturnCode.ExactSolutionRight)
    end

    if sign(fl) == sign(fr)
        @SciMLMessage(
            "The interval is not an enclosing interval, opposite signs at the \
        boundaries are required.",
            verbose, :non_enclosing_interval
        )
        return build_bracketing_solution(prob, alg, left, fl, left, right, ReturnCode.InitialFailure)
    end

    return newton_bracket(
        prob, alg, fun, left, right, fl, fr, zero(fl), zero(fr), false, false, abstol, maxiters
    )
end

# Newton steps safeguarded by bisection on the bracket `[left, right]`. `known_l` and
# `known_r` tell whether `dfl` and `dfr` hold the derivatives at the ends.
function newton_bracket(
        prob, alg, fun, left, right, fl, fr, dfl, dfr, known_l, known_r, abstol, maxiters
    )
    last_step = older_step = oftype(right - left, Inf)
    base_step = oftype(right - left, Inf)   # last plain Newton step from the same end
    base_left = true
    k = 1   # distance beyond the Newton point, in tolerances, once Newton has converged
    for _ in 1:maxiters
        if nextfloat(left) == right
            return newton_bisection_solution(prob, alg, left, right, fl, fr, ReturnCode.FloatingPointLimit)
        end
        if (right - left) / 2 < abstol
            return newton_bisection_solution(prob, alg, left, right, fl, fr, ReturnCode.Success)
        end

        at_left = abs(fl) <= abs(fr)
        if at_left && !known_l
            dfl, known_l = derivative(fun, left), true
        elseif !at_left && !known_r
            dfr, known_r = derivative(fun, right), true
        end
        x, fx, dfx = at_left ? (left, fl, dfl) : (right, fr, dfr)
        step = fx / dfx
        newton = oftype(x, x - step)
        if at_left == base_left && abs(step) < base_step
            ρ = abs(step) / base_step
            if 2ρ > 1
                accelerated = oftype(x, x - min(round(inv(1 - ρ)), oftype(ρ, 16)) * step)
                left < accelerated < right && (newton = accelerated)
            end
        end
        base_left, base_step = at_left, abs(step)

        δ = abs(newton - x)
        if isfinite(newton) && δ <= max(abstol, 16 * eps(x))
            # `x` is at the root up to the tolerance: close the other end of the bracket,
            # doubling the distance until the sign of `f` changes.
            δx = k * max(abstol, eps(x))
            t = at_left ? max(newton, x) + δx : min(newton, x) - δx
            t = clamp(t, nextfloat(left), prevfloat(right))
            k *= 2
        elseif left < newton < right && 2δ <= older_step
            t = newton
        else
            t = (left + right) / 2
        end
        older_step, last_step = last_step, abs(t - x)

        ft = value(fun, t)
        iszero(ft) && return build_exact_solution(prob, alg, t, ft, ReturnCode.Success)
        if sign(ft) == sign(fl)
            left, fl, known_l = t, ft, false
            at_left || (k = 1)
        else
            right, fr, known_r = t, ft, false
            at_left && (k = 1)
        end
    end

    return newton_bisection_solution(prob, alg, left, right, fl, fr, ReturnCode.MaxIters)
end

# The end of the bracket with the smaller residual is the approximate root.
function newton_bisection_solution(prob, alg, left, right, fl, fr, retcode)
    u, fu = abs(fl) <= abs(fr) ? (left, fl) : (right, fr)
    return build_bracketing_solution(prob, alg, u, fu, left, right, retcode)
end
