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

Each iteration evaluates `f` and its derivative once. It takes the Newton step from the
last point, and bisects the bracket instead when that step leaves the bracket or is not
shorter than half the step before the previous one, as `rtsafe` of Numerical Recipes does.
Once the Newton step is within the tolerance, the next point is placed just beyond the
Newton point, on the other side of the root, so that both ends of the bracket converge.

Near a simple root the convergence is quadratic. The method pays off when the derivative
is cheap compared with `f`, for example when both reuse the same expensive computation.
"""
struct NewtonBisection <: AbstractBracketingAlgorithm end

function SciMLBase.__solve(
        prob::IntervalNonlinearProblem, alg::NewtonBisection, args...;
        maxiters = 1000, abstol = nothing, verbose::NonlinearVerbosity = NonlinearVerbosity(), kwargs...
    )
    @assert !SciMLBase.isinplace(prob) "`NewtonBisection` only supports out-of-place problems."
    SciMLBase.has_jac(prob.f) || throw(
        ArgumentError(
            "`NewtonBisection` needs the derivative of `f`, given as \
            `IntervalNonlinearFunction(f; jac)`."
        )
    )

    f = Base.Fix2(prob.f, prob.p)
    df = Base.Fix2(prob.f.jac, prob.p)
    left, right = minmax(promote(prob.tspan...)...)
    fl, fr = f(left), f(right)

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

    # Newton steps start from the end with the shorter step. The last point `x` is always
    # an end of the bracket.
    dfl, dfr = df(left), df(right)
    x, fx, dfx = abs(fl / dfl) <= abs(fr / dfr) ? (left, fl, dfl) : (right, fr, dfr)
    last_step = older_step = oftype(right - left, Inf)
    k = 1   # distance beyond the Newton point, in tolerances, once Newton has converged
    for _ in 1:maxiters
        if nextfloat(left) == right
            return newton_bisection_solution(prob, alg, left, right, fl, fr, ReturnCode.FloatingPointLimit)
        end
        if (right - left) / 2 < abstol
            return newton_bisection_solution(prob, alg, left, right, fl, fr, ReturnCode.Success)
        end

        newton = x - fx / dfx
        δ = abs(newton - x)
        if isfinite(newton) && δ <= max(abstol, 16 * eps(x))
            # `x` is at the root up to the tolerance: close the other end of the bracket,
            # doubling the distance until the sign of `f` changes.
            δx = k * max(abstol, eps(x))
            t = x == left ? max(newton, x) + δx : min(newton, x) - δx
            t = clamp(t, nextfloat(left), prevfloat(right))
            k *= 2
        elseif left < newton < right && 2δ <= older_step
            t = newton
        else
            t = (left + right) / 2
        end
        older_step, last_step = last_step, abs(t - x)

        ft = f(t)
        iszero(ft) && return build_exact_solution(prob, alg, t, ft, ReturnCode.Success)
        was_left = x == left
        if sign(ft) == sign(fl)
            left, fl = t, ft
        else
            right, fr = t, ft
        end
        (t == left) == was_left || (k = 1)
        x, fx, dfx = t, ft, df(t)
    end

    return newton_bisection_solution(prob, alg, left, right, fl, fr, ReturnCode.MaxIters)
end

# The end of the bracket with the smaller residual is the approximate root.
function newton_bisection_solution(prob, alg, left, right, fl, fr, retcode)
    u, fu = abs(fl) <= abs(fr) ? (left, fl) : (right, fr)
    return build_bracketing_solution(prob, alg, u, fu, left, right, retcode)
end
